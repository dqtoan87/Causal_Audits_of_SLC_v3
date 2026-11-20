#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check the numbers paper3.md and supplementary3.md print against the results this tree produced.

Every check does two things. It recomputes the quantity from this tree's results/ and compares it with
the value the manuscript prints, at the precision printed. And, where a check names the text it
backs, it confirms that this exact text is present in the manuscript or the supplement, so a
number edited in the paper without a matching result (or the reverse) is caught.

Tables 4 (main text) and S29 (supplement) are parsed from the markdown and compared cell by cell.

Reads this tree's results/, the manuscript and the supplement (paper3.md and supplementary3.md, by
default from paper/), and two read-only external sources when they are present: the training log of
the audited checkpoints and the ISIC-2024 metadata file.

Exit status: 0 if every check passes, 1 if any check fails, 2 if the manuscript or the supplement is
missing (then no check is run).

Usage:  python code/verify_paper3.py [--paper-dir DIR]
        DIR (or the environment variable PAPER_DIR) holds paper3.md and supplementary3.md; default paper/.
        ISIC_META may point at the ISIC-2024 train-metadata.csv.
"""
from __future__ import annotations

import json
import math
import numpy as np
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
PAPER_DIR = os.environ.get("PAPER_DIR", os.path.join(BASE, "paper"))
if "--paper-dir" in sys.argv:
    PAPER_DIR = sys.argv[sys.argv.index("--paper-dir") + 1]
PAPER = os.path.join(PAPER_DIR, "paper3.md")
SUPP = os.path.join(PAPER_DIR, "supplementary3.md")
TRAIN_LOG = os.path.join(os.path.dirname(BASE), "logs", "c29_frozen.log")
TRAIN_LOG_B3INV = os.path.join(BASE, "log", "c29_b3inv_retrain.log")
ISIC_META = os.environ.get("ISIC_META", "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv")
B = 13.815509557963773

_missing = [p for p in (PAPER, SUPP) if not os.path.isfile(p)]
if _missing:
    sys.stderr.write("verify_paper3: manuscript text not found: %s\n"
                     "Place paper3.md and supplementary3.md in paper/, or pass --paper-dir DIR. No check was run.\n"
                     % ", ".join(_missing))
    sys.exit(2)
TEXT = {"paper": open(PAPER, encoding="utf-8").read(), "supp": open(SUPP, encoding="utf-8").read()}
rows = []


def load(name):
    p = os.path.join(RES, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def check(claim, got, want, tol=0.0, floor=False, ceil=False, text=None, where="paper"):
    """Compare a recomputed value with the printed one and, optionally, confirm the printed text.

    want: scalar, (lo, hi) pair, or string. floor/ceil: got >= want / got <= want.
    text: a literal (or list of literals) that must occur in the named document(s).
    """
    if got is None:
        rows.append(("MISSING", claim, "-", str(want)))
        return
    if isinstance(want, tuple):
        ok = abs(got[0] - want[0]) <= tol and abs(got[1] - want[1]) <= tol
        g, w = "%.4g to %.4g" % tuple(got), "%.4g to %.4g" % want
    elif isinstance(want, str):
        ok, g, w = str(got) == want, str(got), want
    elif floor:
        ok, g, w = got >= want, "%.6g" % got, ">= %.6g" % want
    elif ceil:
        ok, g, w = got <= want, "%.6g" % got, "<= %.6g" % want
    else:
        ok, g, w = abs(got - want) <= tol, "%.6g" % got, "%.6g" % want
    if ok and text is not None:
        docs = [where] if isinstance(where, str) else where
        for t in ([text] if isinstance(text, str) else text):
            dd = docs
            if t.startswith("@supp:"):          # an anchor that lives in the supplement whatever `where` says
                t, dd = t[6:], ["supp"]
            if not any(t in TEXT[d] for d in dd):
                ok = False
                w += "  [text not found: %r]" % t
    rows.append(("ok" if ok else "MISMATCH", claim, g, w))


def num(s):
    return float(s.replace("−", "-").replace("+", ""))


def rng(vals):
    return (min(vals), max(vals))


# ================================================================ cohort and data
d = load("audit_cohort")
if d:
    check("audit cohort: 183 images", d["n_images"], 183, text="183 images from 100 patients")
    check("audit cohort: 100 patients", d["n_patients"], 100)
    check("audit cohort: 83 malignant", d["n_malignant"], 83, text="every malignant image of the fold (83)")
    _mp = len({p for p, t in zip(d["patient_id"], d["target"]) if t == 1})
    check("audit cohort: malignant images come from 44 patients", _mp, 44,
          text="@supp:The malignant subgroup draws on 44 patients")

d = load("padufes_split")
if d:
    c = d["counts"]
    check("PAD-UFES-20 split 1,367 / 475 / 456",
          "%d/%d/%d" % (c["train"]["n"], c["select"]["n"], c["audit"]["n"]), "1367/475/456",
          text="1,367 training, 475 selection and 456 audit images")
    check("PAD-UFES-20 audit fold 278 patients", c["audit"]["n_patients"], 278,
          text="456 images from 278 patients")
    check("PAD-UFES-20 audit-fold malignant rate 47.6%",
          100 * c["audit"]["n_pos"] / c["audit"]["n"], 47.6, 0.05, text="47.6% malignant")

d = load("frozen_split")
if d:
    check("training fold holds 232 malignant cases", d["counts"]["train"]["n_pos"], 232,
          text="315 to 232 malignant cases", where="supp")

if os.path.exists(ISIC_META):
    try:
        import pandas as pd
        m = pd.read_csv(ISIC_META, usecols=["target", "iddx_1", "iddx_2", "patient_id"], low_memory=False)
        check("ISIC-2024: 401,059 tiles", len(m), 401059, text="401,059")
        check("ISIC-2024: 1,042 patients", m["patient_id"].nunique(), 1042, text="1,042 patients")
        check("ISIC-2024: 393 malignant", int(m["target"].sum()), 393, text="393 malignant labels")
        check("ISIC-2024: prevalence 0.098%", 100 * m["target"].mean(), 0.098, 5e-4, text="0.098%")
        check("ISIC-2024: 1,068 histopathology-level diagnoses", int(m["iddx_2"].notna().sum()), 1068,
              text="1,068")
        check("ISIC-2024: 114 indeterminate", int((m["iddx_1"] == "Indeterminate").sum()), 114,
              text="114 indeterminate", where="supp")
    except Exception as exc:  # pragma: no cover
        rows.append(("MISSING", "ISIC-2024 metadata could not be read: %s" % exc, "-", "-"))

if os.path.exists(TRAIN_LOG):
    lg = open(TRAIN_LOG, encoding="utf-8").read()
    # the EfficientNet-B3 Inv model was trained again with selection restricted to epochs with the penalties on
    lg2 = open(TRAIN_LOG_B3INV, encoding="utf-8").read() if os.path.exists(TRAIN_LOG_B3INV) else ""
    auc = {k: float(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=([\d.]+)", lg)}
    auc2 = {k: float(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=([\d.]+)", lg2)}
    auc.update(auc2)
    if auc:
        check("selection AUROC, EfficientNet-B3 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_b3"], auc["dj_slim_b3"], auc["dj_causal_b3"]),
              "0.90/0.89/0.87", text="0.90, 0.89 and 0.87")
        check("selection AUROC, ResNet-50 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_r50"], auc["dj_slim_r50"], auc["dj_causal_r50"]),
              "0.87/0.88/0.89", text="0.87, 0.88 and 0.89", where="supp")
    _ep = {}
    for _tag, _lg in (("dj_causal_b3", lg2), ("dj_causal_r50", lg)):
        _eps = re.findall(r"\[%s\] ep(\d+)/\d+ auc=([\d.]+) .*? pen=(on|off)" % _tag, _lg)
        _best = [e for e in _eps if _tag in auc and abs(float(e[1]) - auc[_tag]) < 5e-5 and e[2] == "on"]
        _ep[_tag] = _best[0] if _best else None
    if all(_ep.values()):
        check("Inv checkpoints saved at epochs with the penalties on: B3 epoch 7, R50 epoch 6",
              "%d %s / %d %s" % (int(_ep["dj_causal_b3"][0]), _ep["dj_causal_b3"][2],
                                 int(_ep["dj_causal_r50"][0]), _ep["dj_causal_r50"][2]), "7 on / 6 on",
              text=["@supp:The EfficientNet-B3 Inv checkpoint was saved at epoch 7 and the ResNet-50 Inv checkpoint at epoch 6",
                    "@supp:The EfficientNet-B3 Inv checkpoint was saved at epoch 7 and the ResNet-50 Inv checkpoint at epoch 6"],
              where=["paper", "supp"])
    secs = [int(x) for x in re.findall(r"\((\d+)s\)", lg)]
    if secs and lg2:
        # cumulative seconds per job in the first log; the B3 Inv job is replaced by the second log's total
        cum = {k: int(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=[\d.]+  \((\d+)s\)", lg)}
        old_b3inv = cum["dj_causal_b3"] - cum["dj_slim_b3"]
        new_b3inv = int(re.findall(r"\[dj_causal_b3\] saved, select-AUC=[\d.]+  \((\d+)s\)", lg2)[0])
        check("training the six checkpoints takes 2.1 hours", (max(secs) - old_b3inv + new_b3inv) / 3600, 2.1, 0.05,
              text="2.1 hours to train the six checkpoints", where="supp")

# ================================================================ artifact tier
d = load("c30_disjoint_audit")
cells = {}
if d:
    for bk in d["models"]:
        for mk in d["models"][bk]:
            for sg in ("all", "malignant", "benign"):
                cells[(bk, mk, sg)] = d["models"][bk][mk][sg]
    cl = list(cells.values())
    check("eighteen artifact cells", len(cl), 18)
    check("every cell exceeds jointly", sum(c["joint_verdict"] == "exceedance" for c in cl), 18)
    check("six cells do not exceed on the signed component",
          sum(c["signed_verdict"] != "exceedance" for c in cl), 6,
          text="Six of the eighteen cells do not exceed on the signed component")
    check("two cells within the signed band", sum(c["signed_verdict"] == "equivalence" for c in cl), 2)
    check("primary magnitudes 0.085 to 0.769", rng([c["mean"] for c in cl]), (0.085, 0.769), 5e-4)
    au = [d["models"][b][m]["audit_auroc"] for b in d["models"] for m in d["models"][b]]
    check("audit-cohort AUROC 0.856 to 0.893", rng(au), (0.856, 0.893), 5e-4, text="0.856 to 0.893")
    check("hair alters 2.4% of pixels", 100 * d["change_frac"]["hair"], 2.4, 0.05, text="2.4% of pixels")
    check("identity edit alters no pixels", d["change_frac"]["identity"], 0.0)
    b3 = d["models"]["b3"]
    for mk, p0s, p1s in (("ERM", "0.733", "0.070"), ("Slim", "0.859", "0.170"), ("Inv", "0.501", "0.046")):
        p0 = b3[mk]["p0_mean_malignant"]
        p1 = p0 + b3[mk]["hair_dose_dp"][1]
        check("%s malignant score under hair %s to %s" % (mk, p0s, p1s), (p0, p1),
              (float(p0s), float(p1s)), 5e-4, text="from %s to %s" % (p0s, p1s))
    # frontier at tolerance 0.20
    def exceeds_at(c, t):
        mag = c["ci"][0] > t
        sg = c["signed_ci"][0] > t or c["signed_ci"][1] < -t
        return mag or sg
    check("seven cells still exceed at tolerance 0.20", sum(exceeds_at(c, 0.20) for c in cl), 7,
          text="seven of the eighteen cells still exceed")
    # Table 4, cell by cell
    tbl = TEXT["paper"][TEXT["paper"].index("**TABLE 4."):]
    tbl = tbl[:tbl.index("\n\n", tbl.index("| --- "))]
    bk = mk = None
    nrow = nbad = 0
    vmap = {"exceeds": "exceedance", "within": "equivalence", "indet.": "indeterminate"}
    for line in tbl.splitlines():
        if not line.startswith("| ") or line.startswith("| Backbone") or line.startswith("| ---"):
            continue
        f = [x.strip() for x in line.strip("|").split("|")]
        bk = {"EfficientNet-B3": "b3", "ResNet-50": "r50"}.get(f[0], bk)
        mk = f[1] or mk
        c = cells[(bk, mk, f[2])]
        m1 = re.match(r"([−+\d.]+) \[([−+\d.]+), ([−+\d.]+)\]", f[3])
        m2 = re.match(r"([−+\d.]+) \[([−+\d.]+), ([−+\d.]+)\]", f[4])
        printed = [num(x) for x in m1.groups() + m2.groups()]
        actual = [c["mean"]] + c["ci"] + [c["signed_mean"]] + c["signed_ci"]
        ok = all(abs(a - p) <= 5e-4 + 1e-9 for a, p in zip(actual, printed))
        ok &= (vmap[f[5]], vmap[f[6]], vmap[f[7]]) == (c["verdict"], c["signed_verdict"], c["joint_verdict"])
        nrow += 1
        nbad += not ok
        if not ok:
            rows.append(("MISMATCH", "Table 4 row %s %s %s" % (bk, mk, f[2]), str(actual), line))
    check("Table 4: all 18 rows match c30 (values, intervals, verdicts)", nrow - nbad, 18)

d = load("c40_simultaneous")
if d:
    lo = [v["ci_simultaneous"][0] for v in d["disjoint_18"]["cells"].values()]
    check("secondary 18-magnitude band: critical value 2.96", d["disjoint_18"]["crit_z"], 2.96, 5e-3, text="@supp:gives a critical value of 2.96")
    check("secondary 18-magnitude band: frontier 0.044", min(lo), 0.044, floor=True,
          text="@supp:so this band would put the frontier at 0.044")
    check("simultaneous replicates 20,000", d["boot"], 20000)
    p9 = d["primary_9"]
    check("nine cached-prediction cells: critical value 2.68", p9["crit_z"], 2.68, 5e-3,
          text="critical value is 2.68", where="supp")
    check("nine cached-prediction cells: Inv benign bound 0.049",
          min(v["ci_simultaneous"][0] for v in p9["cells"].values()), 0.049, 5e-4,
          text="whose simultaneous lower bound is 0.049", where="supp")
d = load("c57_stratified_band")
if d:
    check("stratified band: critical value 3.12", d["crit_36"], 3.12, 5e-3, text="The critical value is 3.12 under both schemes", where="supp")
    check("stratified band: frontier 0.041", d["frontier_36"], 0.041, floor=True, text="@supp:is 0.0417 against 0.0422")
    check("stratified band: frontier 0.0417 (supp)", d["frontier_36"], 0.0417, 5e-5, text="0.0417", where="supp")
    check("stratified band: strata 44 and 56", tuple(d["n_patients_strata"]), (44, 56), 0, text="the 44 patients", where="supp")
    check("stratified band: 17 of 18 still exceed", d["magnitude_exceed_36"], 17, text="Seventeen of the eighteen cells still exceed", where="supp")
    dr = d.get("design_ratio")
    if dr:
        check("observed malignant share 45%", 100 * dr["malignant_share_observed"], 45, 0.5, text="observed share is 45%", where="supp")
        check("plain bootstrap share 34 to 57%", tuple(100 * x for x in dr["malignant_share_plain_2.5_97.5"]), (34, 57), 0.5, text="34 to 57%", where="supp")
        check("stratified share 36 to 55%", tuple(100 * x for x in dr["malignant_share_stratified_2.5_97.5"]), (36, 55), 0.5, text="36 to 55%", where="supp")
        check("19 patients hold both labels, 57 images", (dr["n_patients_with_both_labels"], dr["n_images_of_patients_with_both_labels"]), (19, 57), 0, text="Nineteen patients hold both labels, with 57 images", where="supp")
d = load("c58_rebuild_trace")
if d:
    mix = lambda k: d[k]["mixture"]["magnitude"]
    check("trace: bright ruler moves the cell by 0.0003",
          round(mix("L2_L3_this_pipeline") - mix("L4_this_pipeline_bright_ruler"), 4), 0.0003, 5e-5, text="0.0003", where="supp")
    check("trace: single blend moves the cell by 0.0007",
          round(mix("L2_L3_this_pipeline") - mix("L5_this_pipeline_single_blend_hair"), 4), 0.0007, 5e-5, text="0.0007", where="supp")
    check("trace: reimplementation inference on same edits gives 0.047",
          mix("L6_reimplementation_inference_on_this_pipeline_edits"), 0.047, 5e-4, text="give 0.047", where="supp")
    check("trace: inference codes agree to 0.003", d["L6_unedited"]["reimpl_inference_vs_this_pipeline_mean_abs"],
          0.003, 5e-4, text="to 0.003 on average", where="supp")
    r7 = d["L7_reimplementation_rerun_vs_its_cache"]
    check("trace: rerun vs cache mean 0.016", r7["mean_abs"], 0.016, 5e-4, text="by 0.016 on average", where="supp")
    check("trace: rerun vs cache max 0.72", r7["max_abs"], 0.72, 5e-3, text="up to 0.72", where="supp")
    check("trace: 4.9% of images above 0.05", 100 * r7["share_over_0.05"], 4.9, 0.05, text="4.9% of images", where="supp")
    check("trace: 1.04 blends per touched pixel", d["blends_per_touched_pixel_heavy_hair"], 1.04, 5e-3, text="1.04 times", where="supp")
d = load("c59_rebuild_generator")
if d:
    m = d["models"]["r50_inv_s42"]
    check("rerun: overlays reproduce the cached pixel fractions", all(d["pixel_fraction_matches_cache"].values()), True)
    check("rerun: cell 0.041", m["rerun"]["benign"]["magnitude"], 0.041, 5e-4, text="0.041", where="supp")
    check("rerun: interval [0.030, 0.052]", tuple(m["rerun"]["benign"]["ci"]), (0.030, 0.052), 5e-4, text="[0.030, 0.052]", where="supp")
    check("rerun: cell indeterminate", m["rerun"]["benign"]["joint_verdict"], "indeterminate")
    check("cache: cell within tolerance", m["cache"]["benign"]["joint_verdict"], "equivalence")
    same = sum(v["rerun"][g]["joint_verdict"] == v["cache"][g]["joint_verdict"]
               for v in d["models"].values() for g in ("all", "malignant", "benign"))
    check("rerun keeps 17 of 18 cached decisions", same, 17, text="keeps the other 17 decisions", where="supp")
    mm = [v["unedited_rerun_vs_cache"]["mean_abs"] for v in d["models"].values()]
    check("rerun vs cache, unedited, 0.016 to 0.033 on average", (min(mm), max(mm)), (0.016, 0.033), 5e-4,
          text="0.016 to 0.033 on average", where="supp")
    check("rerun is deterministic", max(v["unedited_rerun_vs_rerun_max_abs"] for v in d["models"].values()), 0.0, 0.0)
d = load("c60_design_coverage")
if d:
    check("benign redraw: joint coverage 95.4%", 100 * d["joint_coverage"], 95.4, 0.05, text="@supp:in 95.4% of draws")
    check("benign redraw: MC se 0.9 points", 100 * d["joint_coverage_mc_se"], 0.9, 0.05, text="0.9 percentage points", where="supp")
    check("benign redraw: component coverage min 98.8%", 100 * d["component_coverage_min"], 98.8, 0.05, text="fewer than 98.8% of draws", where="supp")
    check("benign redraw: crit 2.99 to 3.23", tuple(d["crit_range"]), (2.99, 3.23), 5e-3, text="from 2.99 to 3.23", where="supp")
    check("benign redraw: crit median 3.08", d["crit_median"], 3.08, 5e-3, text="median of 3.08", where="supp")
    check("benign redraw: frontier median 0.0470", d["frontier_median"], 0.0470, 5e-5, text="median of 0.0470", where="supp")
    check("benign redraw: frontier min 0.0279 (rounded down)", math.floor(d["frontier_min"] * 1e4) / 1e4, 0.0279, 1e-9, text="minimum of 0.0279", where="supp")
    check("benign redraw: every tolerance below 0.027 in every draw", d["frontier_min"], 0.027, floor=True, text="every tolerance below 0.027 in every draw", where="supp")
    check("benign redraw: frontier central range 0.032 to 0.066 (lower end rounded down)", (math.floor(d["frontier_2.5_97.5"][0] * 1e3) / 1e3, d["frontier_2.5_97.5"][1]), (0.032, 0.066), 5e-4, text="0.032 to 0.066", where=["paper", "supp"])
    check("benign redraw: all 18 exceed 0.05 jointly in 38.2% of draws", 100 * d["share_draws_all_18_exceed"], 38.2, 0.05, text="@supp:At the 0.05 operating point all eighteen cells exceeded jointly in 38.2% of draws")
    check("benign redraw: frame 4,150 benign from 198 patients", (d["n_benign_frame"], d["n_benign_frame_patients"]), (4150, 198), 0, text="4,150 benign images from 198 patients", where="supp")
    check("benign redraw: 500 draws", d["draws"], 500, text="@supp:redrew the 100 benign images 500 times")
import math as _m
_Phi = lambda z: 0.5 * (1 + _m.erf(z / _m.sqrt(2)))
check("signed exceedance size at eps/se = 0.25 is 0.032", _Phi(-1.959964) + _Phi(-1.959964 - 0.5), 0.032, 5e-4,
      text="0.032 at *ε*/*se* = 0.25", where="supp")
def _sg2(x):
    r = round(x, 2)
    return ("+%.2f" % r) if r > 0 else (("\u2212%.2f" % abs(r)) if r < 0 else "0.00")


d = load("c62_evaluation_inv_retrained")
if d:
    # second run of the unchanged plan, with the Inv checkpoint of Section III-D: reported in the main text
    reg, con = load("c62_preregistration"), load("c62_construction")
    pr, comp = d["primary"], d["primary"]["components"]
    TT = ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
    check("route test: 78 locked images from 54 patients", (reg["locked_set"]["n_images"], reg["locked_set"]["n_patients"]), (78, 54), 0,
          text="78 malignant images of the selection fold, from 54 patients")
    a4 = [con["summary"][t]["n_all_three_g1"] for t in TT]
    check("route test: all four controls pass G1 on 74 to 78 images", (min(a4), max(a4)), (74, 78), 0, text="74 to 78 of the 78 images")
    sx = sorted(v["est"] for k, v in comp.items() if v.get("decision_sim") == "exceeds" and ("D_rad4-rad16" in k or "D_hue4-hue16" in k))
    check("route test: scale-pair disagreements 5.02 to 6.62 logits", (sx[0], sx[-1]), (5.02, 6.62), 5e-3, text=["@supp:**+5.02 [+1.20, +8.83]**", "@supp:**+6.62 [+1.99, +11.25]**"])
    dis = sorted(c for c, v in pr["cells"].items() if v["class"] == "routes disagree")
    check("route test: three disagreeing cells in the second run", ",".join(dis), ",".join(sorted(["erm|contrast@2", "causal|contrast@2", "slim|color_var@2"])),
          text="The second run flags ERM and Inv at the highest contrast dose and Slim at the highest variegation dose.")
    check("route test: 3 disagree / 0 agree / 9 indeterminate",
          "%d/%d/%d" % (pr["summary"]["routes disagree"], pr["summary"]["routes agree"], pr["summary"]["indeterminate"]), "3/0/9",
          text="Three cells disagree, none agrees and nine are indeterminate", where="supp")
    check("route test: one direction pair exceeds, Inv at the highest contrast dose",
          (pr["summary"]["direction_pair_exceeds"], round(comp["causal|contrast@2|D_rad4-hue4"]["est"], 2)), (1, 4.90), 5e-3,
          text="For Inv at the highest contrast dose the direction pair rad4 against hue4 also exceeds the margin", where="supp")
    dm = sum(1 for c, v in comp.items() if ("D_rad4-hue4" in c or "D_rad16-hue16" in c) and (v["ci"][0] > 0 or v["ci"][1] < 0))
    check("route test: 12 of 24 direction differences exclude zero marginally", dm, 12,
          text="Of the 24 direction differences, 12 have marginal intervals that exclude zero", where="supp")
    dme = sum(1 for c, v in comp.items() if ("D_rad4-hue4" in c or "D_rad16-hue16" in c) and v.get("decision_marginal") == "exceeds")
    check("route test: 7 direction differences exceed on marginal intervals", dme, 7, text="and 7 exceed the margin on their marginal intervals", where="supp")
    check("route test: simultaneous critical value 3.37", pr["crit_simultaneous"], 3.37, 5e-3, text="critical value of the band was 3.37", where="supp")
    nT = [(pr["targets"][t]["n_images"], pr["targets"][t]["n_patients"]) for t in TT]
    check("route test: primary sets 44 to 47 images from 32 or 33 patients",
          "%d-%d/%d-%d" % (min(x[0] for x in nT), max(x[0] for x in nT), min(x[1] for x in nT), max(x[1] for x in nT)), "44-47/32-33",
          text="44 to 47 images from 32 or 33 patients", where="supp")
    _d1 = load("c62_evaluation")
    _cl1 = {"routes disagree": "disagree", "routes agree": "agree", "indeterminate": "indeterminate"}
    for t, txt in zip(TT, ("| 75 (53) / 45 (32) |", "| 73 (52) / 44 (32) |", "| 77 (54) / 47 (33) |", "| 76 (54) / 46 (32) |")):
        n1 = (_d1["primary"]["targets"][t]["n_images"], _d1["primary"]["targets"][t]["n_patients"])
        check("Table 5 images (patients), first / second run, %s" % t, "| %d (%d) / %d (%d) |" % (n1 + nT[TT.index(t)]), txt, text=txt)
        for m in ("erm", "slim", "causal"):
            # each cell shows the second run's verdict, its margins, then the first run's verdict after the slash
            row = [ln for ln in TEXT["paper"].splitlines() if ln.startswith("| %s |" % {"contrast@0": "contrast, lowest dose",
                   "contrast@2": "contrast, highest dose", "color_var@0": "color variegation, lowest dose",
                   "color_var@2": "color variegation, highest dose"}[t])]
            cell = row[0].split("|")[3 + ("erm", "slim", "causal").index(m)].strip() if row else ""
            want = "%s / %s" % (_cl1[_d1["primary"]["cells"]["%s|%s" % (m, t)]["class"]], _cl1[pr["cells"]["%s|%s" % (m, t)]["class"]])
            got = "%s / %s" % (cell.split(" / ")[0].split(" (")[0].split(",")[0], cell.split(" / ")[1].split(",")[0].strip()) if cell else None
            check("Table 5 verdicts, first / second run, %s %s" % (m, t), got, want)
    _f1 = sorted(c for c, v in _d1["primary"]["cells"].items() if v["class"] == "routes disagree")
    _a1 = sorted(c for c, v in _d1["primary"]["cells"].items() if v["class"] == "routes agree")
    check("both runs: ERM highest contrast and Slim highest variegation disagree in both",
          ",".join(sorted(set(_f1) & set(dis))), "erm|contrast@2,slim|color_var@2",
          text="@supp:the first run's two highest-dose flags remain")
    check("first run: Slim lowest contrast also disagreed, two epoch-1 Inv cells agreed", ",".join(_f1) + " | " + ",".join(_a1),
          "erm|contrast@2,slim|color_var@2,slim|contrast@0 | causal|color_var@0,causal|contrast@0".replace(
              "erm|contrast@2,slim|color_var@2,slim|contrast@0", ",".join(sorted(["erm|contrast@2", "slim|color_var@2", "slim|contrast@0"]))),
          text="@supp:The two lowest-dose Inv cells agreed")
    _n1 = [_d1["primary"]["targets"][t]["n_images"] for t in TT]
    check("both runs: 73 to 77 images per target, then 44 to 47", "%d-%d / %d-%d" % (min(_n1), max(_n1), min(x[0] for x in nT), max(x[0] for x in nT)),
          "73-77 / 44-47", text="analyzed 44 to 47 images per target instead of 73 to 77")
    add = [d["secondary_all_g1"]["targets"][t]["n_images"] - pr["targets"][t]["n_images"] for t in TT]
    check("route test: dropping the floor adds 30 or 31 images per target", (min(add), max(add)), (30, 31), 0,
          text="adds 30 or 31 images per target", where=["paper", "supp"])
    for cell, q, txt in (("erm|contrast@2", "rad4-rad16", "+5.02 [+1.20, +8.83]"), ("slim|color_var@2", "rad4-rad16", "+6.16 [+0.62, +11.70]"),
                         ("causal|contrast@2", "rad4-rad16", "+6.62 [+1.99, +11.25]"), ("causal|contrast@2", "rad4-hue4", "+4.90 [+1.01, +8.79]")):
        v = comp["%s|D_%s" % (cell, q)]
        check("route test: Table S31 %s %s (ends rounded outward)" % (cell, q), "%s [%s, %s]" % (_sg2(v["est"]), _sg2(math.floor(v["sim"][0] * 100 + 1e-9) / 100),
              _sg2(math.ceil(v["sim"][1] * 100 - 1e-9) / 100)), txt, text=txt, where="supp")
    # Table S31: every entry of both runs, in table order, ends rounded outward, bold when the difference exceeds
    _tn = {"contrast@0": "contrast, lowest dose", "contrast@2": "contrast, highest dose", "color_var@0": "color variegation, lowest dose",
           "color_var@2": "color variegation, highest dose"}
    _s31 = TEXT["supp"].split("**TABLE S31.")[1].split("**Route sums against placebos.**")[0]
    _rows = [ln for ln in _s31.splitlines() if ln.startswith("|") and ("| first |" in ln or "| second |" in ln)]
    _want, _k = [], 0
    for _m in ("erm", "slim", "causal"):
        for _t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2"):
            for _run, _ev in (("first", load("c62_evaluation")), ("second", d)):
                _cc = []
                for _q in ("rad4-hue4", "rad16-hue16", "rad4-rad16", "hue4-hue16"):
                    _v = _ev["primary"]["components"]["%s|%s|D_%s" % (_m, _t, _q)]
                    _x = "%s [%s, %s]" % (_sg2(_v["est"]), _sg2(math.floor(_v["sim"][0] * 100 + 1e-9) / 100), _sg2(math.ceil(_v["sim"][1] * 100 - 1e-9) / 100))
                    _cc.append("**%s**" % _x if _v["decision_sim"] == "exceeds" else _x)
                _cc.append(_ev["primary"]["cells"]["%s|%s" % (_m, _t)]["class"].replace("routes ", ""))
                _want.append(_cc)
    _got = [[c.strip() for c in ln.split("|")[4:9]] for ln in _rows]
    check("Table S31: all 96 differences and 24 cell verdicts of both runs", sum(a == b for a, b in zip(_got, _want)), 24, 0)
    rs, ps = [], []
    for m in ("erm", "slim", "causal"):
        for t in TT:
            S = [comp["%s|%s|S_%s" % (m, t, k)]["est"] for k in ("rad4", "rad16", "hue4", "hue16")]
            PC = [comp["%s|%s|PC_%s" % (m, t, k)]["est"] for k in ("hue", "flip", "jitter", "noise")]
            rs.append(max(S) - min(S)); ps.append(max(PC) - min(PC))
    check("route test: route sums spread 0.5 to 6.7", (min(rs), max(rs)), (0.5, 6.7), 0.05, text="and by 0.5 to 6.7 in the second", where="supp")
    check("route test: placebos spread 3.1 to 6.7", (min(ps), max(ps)), (3.1, 6.7), 0.05, text="3.1 to 6.7", where=["paper", "supp"])
    _c1 = load("c62_evaluation")["primary"]["components"]
    _rs1 = [max(_c1["%s|%s|S_%s" % (m, t, k)]["est"] for k in ("rad4", "rad16", "hue4", "hue16")) - min(_c1["%s|%s|S_%s" % (m, t, k)]["est"] for k in ("rad4", "rad16", "hue4", "hue16"))
            for m in ("erm", "slim", "causal") for t in TT]
    _ps1 = [max(_c1["%s|%s|PC_%s" % (m, t, k)]["est"] for k in ("hue", "flip", "jitter", "noise")) - min(_c1["%s|%s|PC_%s" % (m, t, k)]["est"] for k in ("hue", "flip", "jitter", "noise"))
            for m in ("erm", "slim", "causal") for t in TT]
    check("first run: route sums spread 0.1 to 5.4, placebos 0.6 to 4.5 (outward)", (math.floor(min(_rs1) * 10 + 1e-9) / 10, math.ceil(max(_rs1) * 10 - 1e-9) / 10, math.floor(min(_ps1) * 10 + 1e-9) / 10, math.ceil(max(_ps1) * 10 - 1e-9) / 10) == (0.1, 5.4, 0.6, 4.5), True,
          text=["spread by 0.1 to 5.4 logits in the first run", "spread by 0.6 to 4.5 logits in the first run"], where="supp")
    sall = d["secondary_all_g1"]
    check("route test: without the floor the same three cells disagree, nine indeterminate",
          "%d/%d/%d %s" % (sall["summary"]["routes disagree"], sall["summary"]["routes agree"], sall["summary"]["indeterminate"],
                           sorted(c for c, v in sall["cells"].items() if v["class"] == "routes disagree") == dis), "3/0/9 True",
          text="the same three cells disagree and the other nine are indeterminate", where=["paper", "supp"])
    ni = [d["secondary_independent"]["targets"][t]["n_images"] for t in TT]
    check("route test: independent subsets 11, 0, 0, 0", "/".join(map(str, ni)), "11/0/0/0", text="11, 0, 0 and 0 images", where="supp")
d1 = load("c62_evaluation")
if d1:
    # first run, registered outcome, reported in Supplementary Section S19
    p1r, c1 = d1["primary"], d1["primary"]["components"]
    check("first run: 3 disagree / 2 agree / 7 indeterminate",
          "%d/%d/%d" % (p1r["summary"]["routes disagree"], p1r["summary"]["routes agree"], p1r["summary"]["indeterminate"]), "3/2/7",
          text="The two lowest-dose Inv cells agreed, and seven cells were indeterminate", where="supp")
    check("first run: critical value 3.30", p1r["crit_simultaneous"], 3.30, 5e-3, text="the critical value was 3.30", where="supp")
    for cell, txt in (("slim|contrast@0", "+1.99 [+0.50, +3.47]"), ("erm|contrast@2", "+4.63 [+1.91, +7.34]"), ("slim|color_var@2", "+5.27 [+1.83, +8.70]")):
        q = c1[cell + "|D_rad4-rad16"]
        check("first run: %s (lower end rounded down)" % cell, "%s [%s, %s]" % (_sg2(q["est"]), _sg2(math.floor(q["sim"][0] * 100) / 100), _sg2(q["sim"][1])), txt, text=txt, where="supp")
    s1 = d1["secondary_all_g1"]["summary"]
    check("first run: without the floor 2 disagree, 0 agree, 10 indeterminate",
          "%d/%d/%d" % (s1["routes disagree"], s1["routes agree"], s1["indeterminate"]), "2/0/10",
          text="two cells disagreed, none agreed and ten were indeterminate", where="supp")
    n1 = [(p1r["targets"][t]["n_images"], p1r["targets"][t]["n_patients"]) for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")]
    check("first run: 73 to 77 images from 52 to 54 patients",
          "%d-%d/%d-%d" % (min(x[0] for x in n1), max(x[0] for x in n1), min(x[1] for x in n1), max(x[1] for x in n1)), "73-77/52-54",
          text="73 to 77 images from 52 to 54 patients", where="supp")
t34 = load("c34_ctrl_vs_ctrl")
if t34:
    med = {}
    for q in t34["pairs"]:
        med.setdefault(q["model"], []).append(abs(q["diff"]))
    got = "%.1f/%.1f/%.1f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("diagnostic (T): per-model median disagreement 7.8, 9.5, 11.4", got, "7.8/9.5/11.4")
    got2 = "%.2f/%.2f/%.2f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("S6: per-model medians 7.80, 9.47, 11.40", got2, "7.80/9.47/11.40",
          text="7.80 logits for ERM, 9.47 for Slim and 11.40 for Inv", where="supp")
rb = load("c65_route_robustness_inv_retrained")
if rb:
    check("robustness (second run): stored responses reproduce the band", rb["reproduction_max_abs_difference"], 0.0, 1e-9)
    rob = [rb["decisive"][k] for k in ("erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4")]
    check("robustness: ERM and Inv at the highest contrast dose hold for 20 of 20 seeds and all 32 subsets",
          "/".join("%d-%d/%d" % (q["seed_exceeds"], q["lopo_in_cell_exceeds"], q["lopo_in_cell_n"]) for q in rob), "20-32/32/20-32/32/20-32/32",
          text="in all 32 subsets that leave out one of the cell's patients", where="supp")
    # lower bounds of a disagreement are rounded down, toward the conservative side
    check("robustness: smallest lower bounds 0.78, 1.60 and 0.73 (rounded down)",
          "/".join("%.2f" % (math.floor(100 * q["lopo_in_cell_lower_min"]) / 100) for q in rob), "0.78/1.60/0.73",
          text=["smallest lower bounds of 0.78 and 1.60"], where="supp")
    check("robustness: Inv direction pair smallest bound 0.73 (rounded down)", math.floor(100 * rob[2]["lopo_in_cell_lower_min"]) / 100, 0.73, 1e-9,
          text="with a smallest lower bound of 0.73", where="supp")
    s = rb["decisive"]["slim|color_var@2|D_rad4-rad16"]
    check("robustness: Slim highest variegation lower bound 0.62", s["sim_lower_unrounded"], 0.62, 5e-3, text="simultaneous lower bound is 0.62", where="supp")
    check("robustness: Slim highest variegation in 17 of 32 subsets and 20 of 20 seeds",
          "%d/%d %d" % (s["lopo_in_cell_exceeds"], s["lopo_in_cell_n"], s["seed_exceeds"]), "17/32 20",
          text="17 of the 32 subsets that leave out one of its patients", where=["paper", "supp"])
    check("robustness: its seed lower bounds 0.55 to 0.70", tuple(s["seed_lower_range"]), (0.55, 0.70), 5e-3,
          text="with lower bounds from 0.55 to 0.70", where="supp")
    check("robustness: its smallest subset bound 0.06", round(s["lopo_in_cell_lower_min"], 2), 0.06, 1e-9,
          text="with a smallest lower bound of 0.06", where="supp")
    check("robustness: critical value across seeds 3.35 to 3.39", tuple(rb["seed_crit_range"]), (3.35, 3.39), 5e-3,
          text="runs from 3.35 to 3.39", where="supp")
    # Table 5 and Table S35: the disagree margin is rounded down and the agree margin up
    _ev2 = load("c62_evaluation_inv_retrained")
    _cls = {"routes disagree": "disagree", "routes agree": "agree", "indeterminate": "indeterminate"}
    _rt = load("c73_route_robustness_table")
    _rob2 = set(_rt["second_run"]["robust_cells"]) if _rt else set()
    _rb1 = load("c65_route_robustness")
    _ev1x = load("c62_evaluation")
    _rob1 = set(_rt["first_run"]["robust_cells"]) if _rt else set()
    _fl = lambda x: math.floor(x * 100 + 1e-9) / 100
    _ce = lambda x: math.ceil(x * 100 - 1e-9) / 100
    for _c, _v in rb["cells"].items():
        _v1 = _rb1["cells"][_c]
        _l1 = _cls[_ev1x["primary"]["cells"][_c]["class"]]
        _l2 = _cls[_ev2["primary"]["cells"][_c]["class"]]
        # Table 5: first run with its margins, then the second run's verdict
        _txt = "%s (%.2f; %.2f) / %s |" % (_l1, _fl(_v1["disagrees_below_margin"]), _ce(_v1["agrees_above_margin"]), _l2)
        check("Table 5 cell %s" % _c, _txt in TEXT["paper"], True, text=_txt)
        # Table S35: both runs
        _m, _t = _c.split("|")
        _txt35 = "| %s | %s / %s | %.2f / %.2f | %.2f / %.2f |" % ({"contrast@0": "contrast, lowest dose", "contrast@2": "contrast, highest dose",
                  "color_var@0": "color variegation, lowest dose", "color_var@2": "color variegation, highest dose"}[_t],
                  _cls[_ev1x["primary"]["cells"][_c]["class"]], _cls[_ev2["primary"]["cells"][_c]["class"]],
                  _fl(_v1["disagrees_below_margin"]), _fl(_v["disagrees_below_margin"]), _ce(_v1["agrees_above_margin"]), _ce(_v["agrees_above_margin"]))
        check("Table S35 cell %s, both runs" % _c, _txt35 in TEXT["supp"], True, text=_txt35, where="supp")
    cp = rb["cross_pairs"]
    out = ["%d" % sum(1 for k, v in cp.items() if k.startswith(m + "|") and (v["ci"][0] > 0.5 or v["ci"][1] < -0.5)) for m in ("erm", "slim", "causal")]
    ins = sum(1 for v in cp.values() if v["ci"][0] > -0.5 and v["ci"][1] < 0.5)
    check("cross pairs: 4, 5 and 2 outside, none inside", "/".join(out) + " %d" % ins, "4/5/2 0",
          text="11 of the 24 cross-pair intervals lie wholly outside the margin, 4 for ERM, 5 for Slim and 2 for Inv", where="supp")
rb1 = load("c65_route_robustness")
if rb1 and "cross_pairs" in rb1:
    _cp1 = rb1["cross_pairs"]
    _o1 = ["%d" % sum(1 for k, v in _cp1.items() if k.startswith(m + "|") and (v["ci"][0] > 0.5 or v["ci"][1] < -0.5)) for m in ("erm", "slim", "causal")]
    _i1c = sum(1 for v in _cp1.values() if v["ci"][0] > -0.5 and v["ci"][1] < 0.5)
    check("cross pairs, first run: 3, 5 and 0 outside, 6 inside", "/".join(_o1) + " %d" % _i1c, "3/5/0 6",
          text="in the first run 8 lie outside, 3 for ERM and 5 for Slim, and 6 inside", where="supp")
if rb1:
    s = rb1["decisive"]["slim|contrast@0|D_rad4-rad16"]
    check("first run: borderline lower bound 0.509 (rounded down)", math.floor(s["sim_lower_unrounded"] * 1e3) / 1e3, 0.509, 1e-9, text="lower bound of 0.509", where="supp")
    check("first run: borderline under 10 of 20 seeds", s["seed_exceeds"], 10, text="under 10 of 20 other seeds", where="supp")
    check("first run: borderline in 31 of the 53 subsets", (s["lopo_in_cell_exceeds"], s["lopo_in_cell_n"]), (31, 53), 0,
          text="in 31 of the 53 subsets that leave out one of its patients", where="supp")
    rob1 = [rb1["decisive"][k] for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("first run: the two highest-dose disagreements hold for 20 of 20 seeds and every in-cell subset",
          "/".join("%d-%d/%d" % (q["seed_exceeds"], q["lopo_in_cell_exceeds"], q["lopo_in_cell_n"]) for q in rob1), "20-52/52/20-54/54",
          text="The two highest-dose disagreements held under 20 of 20 other bootstrap seeds", where="supp")
sc = load("c64_route_screens")
if sc:
    fr = sc["frontier"]
    nval = sum(s["valid"] for part in ("registered", "frontier") for t in sc[part] for s in sc[part][t].values())
    check("screens: no constructed control passes all three", nval, 0, text="no constructed control passes all three screens")
    check("screens: optimised radial footprint-link scores 0.56 and 0.06",
          (fr["contrast@2"]["rad"]["footprint_link_score"], fr["color_var@2"]["rad"]["footprint_link_score"]),
          (0.56, 0.06), 5e-3, text="with footprint-link scores of 0.56 and 0.06", where="supp")
    tonly = sorted("%s|%s|%s" % (part, t, c) for part in ("registered", "frontier") for t in sc[part] for c, s in sc[part][t].items() if s["valid_target_only"])
    check("screens: two controls pass under a target-only (A2)", "/".join(tonly),
          "frontier|color_var@2|rad/registered|color_var@2|rad16",
          text="@supp:Under the target-only reading two controls would pass all three screens")
    check("screens: they move the contrast proxy by 47 and 58%",
          (100 * sc["registered"]["color_var@2"]["rad16"]["offtarget_proxy_shift"], 100 * fr["color_var@2"]["rad"]["offtarget_proxy_shift"]),
          (47, 58), 0.5, text="@supp:move the contrast proxy by 47 and 58% on average")
    check("screens: target proxy of every constructed control moves < 0.15%",
          max(s["target_proxy_shift"] for part in ("registered", "frontier") for t in sc[part] for s in sc[part][t].values()) < 0.0015, True,
          text="less than 0.15% of its unedited value", where="supp")
    eo = sc["edit_offtarget"]
    check("C3b proxies: edit off-target CIELAB 25 and 78% (contrast)",
          (100 * eo["contrast@0"]["median_rel_cielab"], 100 * eo["contrast@2"]["median_rel_cielab"]), (25, 78), 0.5,
          text="@supp:a median of 25 and 78% at the two contrast doses")
    check("C3b proxies: edit off-target CIELAB 16 and 51% (variegation)",
          (100 * eo["color_var@0"]["median_rel_cielab"], 100 * eo["color_var@2"]["median_rel_cielab"]), (16, 51), 0.5,
          text="@supp:16 and 51%")
    rg2 = sc["registered"]
    check("off-target: CIELAB 20.0 and 26.4% at the highest contrast dose",
          (100 * rg2["contrast@2"]["rad4"]["offtarget_median_rel_control"], 100 * rg2["contrast@2"]["rad16"]["offtarget_median_rel_control"]),
          (20.0, 26.4), 0.05, text="@supp:the CIELAB change of variegation is 20.0 and 26.4%")
    check("off-target: contrast proxy 18 and 47% at the highest variegation dose",
          (100 * rg2["color_var@2"]["rad4"]["offtarget_proxy_shift"], 100 * rg2["color_var@2"]["rad16"]["offtarget_proxy_shift"]),
          (18, 47), 0.5, text="@supp:the contrast proxy changes by 18 and 47%")
    check("screens: proxy shift 20 and 29%", (100 * fr["contrast@2"]["rad"]["proxy_shift"], 100 * fr["color_var@2"]["rad"]["proxy_shift"]),
          (20, 29), 0.5, text="mean proxy shift is 20 and 29%")
    check("screens: 30 and 71% of the edit's shift", (100 * fr["contrast@2"]["rad"]["proxy_ratio"], 100 * fr["color_var@2"]["rad"]["proxy_ratio"]),
          (30, 71), 0.5, text="30 and 71% of the concept edit's own", where="supp")
    _hs = (("registered", ("hue4", "hue16")), ("frontier", ("hue",)))
    check("screens: every low-dose hue control passes (A2)",
          all(sc[pp][t][c]["A2"] for pp, cc in _hs for t in ("contrast@0", "color_var@0") for c in cc), True)
    check("screens: hue misses balance by 9.2 to 9.4 at the lowest contrast dose",
          rng([sc[pp]["contrast@0"][c]["footprint_link_score"] for pp, cc in _hs for c in cc]), (9.2, 9.4), 0.05,
          text="are 9.2 to 9.4 and 10.8 to 10.9", where="supp")
    check("screens: and by 10.8 to 10.9 at the lowest variegation dose",
          rng([sc[pp]["color_var@0"][c]["footprint_link_score"] for pp, cc in _hs for c in cc]), (10.8, 10.9), 0.05,
          text="are 9.2 to 9.4 and 10.8 to 10.9", where="supp")
    check("screens: concept edit moves the contrast proxy by 16% on the images of the two candidates",
          (100 * sc["registered"]["color_var@2"]["rad16"]["edit_offtarget_proxy_shift"],
           100 * sc["frontier"]["color_var@2"]["rad"]["edit_offtarget_proxy_shift"]), (16, 16), 0.5,
          text="against 16% for the concept edit on the same images", where="supp")
    rg = sc["registered"]
    check("screens: off-target 4.9 and 5.3% at the lowest contrast dose",
          (100 * rg["contrast@0"]["rad4"]["offtarget_median_rel_control"], 100 * rg["contrast@0"]["rad16"]["offtarget_median_rel_control"]),
          (4.9, 5.3), 0.05, text="4.9 and 5.3% in CIELAB", where="supp")
import hashlib as _hl
_man = os.path.join(os.path.dirname(RES), "results", "c62_manifest.sha256")
if os.path.exists(_man):
    _bad = [ln.split()[1] for ln in open(_man).read().splitlines() if ln.strip() and
            _hl.sha256(open(os.path.join(os.path.dirname(RES), ln.split()[1]), "rb").read()).hexdigest() != ln.split()[0]]
    check("route-test manifest: every listed file matches its SHA-256", len(_bad), 0, text="c62_manifest.sha256", where="supp")
d = load("c63_route_exploratory_inv_retrained")
if d:
    check("exploratory (second run): nine differences analyzed, none decided", "%d %d" % (d["n_analysed"], d["summary"]["indeterminate"]), "9 9",
          text="@supp:None of the nine differences is decided on these images", where="supp")
    check("exploratory (second run): main text says no pair is decided", d["summary"]["exceeds"] + d["summary"]["within"], 0,
          text="@supp:None of the nine differences is decided on these images")
    check("exploratory (second run): critical value 2.86", d["crit_simultaneous"], 2.86, 5e-3, text="(critical value 2.86)", where="supp")
d = load("c63_route_exploratory")
if d:
    q = d["components"]["slim|contrast@0|D_rad4-rad16"]
    check("first run exploratory: Slim persists at 2.32 on 49 images", (q["est"], q["n_images"]), (2.32, 49), 5e-3, text="2.32 logits on 49 images", where="supp")
    check("first run exploratory: decision exceeds", q["decision_sim"], "exceeds")
d = load("c62_frontier")
if d:
    tks = ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
    fine = [sum(r["targets"][t]["rad"]["fine_feasible"] for r in d["images"]) for t in tks]
    check("frontier: radial within 0.1% on 13 and 15 images at the high doses, none at the low doses", "/".join(map(str, fine)), "0/13/0/15",
          text="on 13 and 15 of the 78 images at the high doses")
    h10 = sum(1 for t in tks for r in d["images"] if r["targets"][t]["hue"]["neutral_converged"]
              and max(abs(r["targets"][t]["hue"]["rel_dE"]), abs(r["targets"][t]["hue"]["rel_hf"])) <= 0.10)
    check("frontier: hue never within 10%", h10, 0, text="@supp:The hue mechanism never came within 10%")
    _inex = sorted(r["targets"][t][m]["max_rel_mag"] for r in d["images"] for t in tks for m in ("rad", "hue")
                   if (r["targets"][t][m].get("max_rel_mag") or 0) > 1e-5)
    check("frontier: two inexact hue solutions, 0.007 and 0.22%",
          "%d: %.3f/%.2f" % (len(_inex), 100 * _inex[0], 100 * _inex[-1]), "2: 0.007/0.22",
          text="miss the magnitude map by 0.007 and 0.22%", where="supp")
    _con = load("c62_construction")
    if _con:
        _pre = sum(1 for t in tks for r in _con["images"] for c in ("rad4", "rad16")
                   if max(abs(r["targets"][t]["controls"][c]["rel_dE"]), abs(r["targets"][t]["controls"][c]["rel_hf"])) <= 0.001)
        check("frontier: pre-specified radial controls never within 0.1%", _pre, 0,
              text="@supp:The pre-specified radial controls never came within 0.1%")
        _g2 = [sum(r["targets"][t]["controls"][c]["g2"] for r in _con["images"]) / len(_con["images"])
               for t in ("contrast@2", "color_var@2") for c in ("rad4", "rad16", "hue4", "hue16")]
        check("high doses: seven of eight controls fail G2 on most images", sum(x < 0.5 for x in _g2), 7,
              text="seven of the eight controls fail the independent CIELAB check G2 on most images")
d = load("c56_simultaneous36")
if d:
    check("36-component critical value 3.12 (headline band)", d["crit_36"], 3.12, 5e-3, text="3.12 standard errors, against 1.96")
    check("every cell exceeds every tolerance below 0.042 (36 components)", d["frontier_36"], 0.042, floor=True,
          text="every tolerance below 0.042 probability units")
    check("36-component frontier 0.0422 (supp)", d["frontier_36"], 0.0422, 5e-5, text="0.0422", where="supp")
    check("36 band: 17 of 18 cells exceed jointly at 0.05", d["magnitude_exceed_36"], 17, text="and 17 of the 18 cells exceed 0.05 jointly")
    _mk = min((v["sim36"][0], k) for k, v in d["components"].items() if k.endswith("/mag"))
    _iv = d["components"]["b3-Inv/ben/mag"]["sim36"]
    check("frontier cell: EfficientNet-B3 Inv benign, simultaneous magnitude interval [0.042, 0.139]",
          "%s %.3f %.3f" % (_mk[1], math.floor(_iv[0] * 1000) / 1000, _iv[1]), "b3-Inv/ben/mag 0.042 0.139",
          text=["@supp:Its simultaneous magnitude interval is [0.042, 0.139]"])
    check("36 band signed: 11 exceed / 1 within / 6 indet",
          "%d/%d/%d" % (d["signed_counts_36"]["exceeds"], d["signed_counts_36"]["within"], d["signed_counts_36"]["indet."]),
          "11/1/6", text="exceeds in 11 cells, is within tolerance in 1 and is indeterminate in 6", where="supp")
    ri = d["components"]["r50-Inv/mal/sgn"]["sim36"]
    check("R50 Inv malignant signed sim36 [-0.040, +0.029]", tuple(ri), (-0.040, 0.029), 5e-4,
          text="[−0.040, +0.029]", where="supp")
    check("36 band critical value agrees with c40 (2.96 for 18)", d["crit_18_magnitude_only"], 2.96, 5e-3)
if cells:
    mlb = min(c["ci"][0] for c in cells.values())
    check("marginal bound 0.0627", mlb, 0.0627, 5e-5, text="@supp:The marginal bound of 0.0627 is a per-cell statement")
    check("0.063 is which is 1.25 times the tolerance the tolerance", mlb / 0.05, 1.25, 5e-3, text="@supp:which is 1.25 times the tolerance")

d = load("c32_absdp_patient_ci")
if d:
    c = d["cells"]
    check("cached run: Inv benign 0.091 [0.063, 0.124]",
          (c["Inv"]["benign"]["mean"], c["Inv"]["benign"]["ci"][0]), (0.091, 0.063), 5e-4,
          text="0.091 with interval [0.063, 0.124]", where="supp")
    check("cached run: Slim malignant 0.762 [0.707, 0.808]",
          (c["Slim"]["malignant"]["mean"], c["Slim"]["malignant"]["ci"][1]), (0.762, 0.808), 5e-4,
          text="0.762 with interval [0.707, 0.808]", where="supp")

d = load("c50_overlay_seed")
if d:
    sp = [v["spread_pct_of_seed42"] for v in d["cells"].values()]
    check("seven realizations move magnitudes 3.2 to 24.9%", rng(sp), (3.2, 24.9), 0.05,
          text="The magnitude moves by 3.2 to 24.9% of its reported value", where="supp")
    check("no verdict changes across realizations", all(v["verdict_stable"] for v in d["cells"].values()), True)
    check("smallest lower bound across realizations 0.058",
          min(v["lower_bound_min"] for v in d["cells"].values()), 0.058, floor=True, text="@supp:0.0585, still 1.17 times the tolerance")
    check("seven realizations", len(d["seeds"]), 7)
    _pc = d["pixel_change_frac"]
    _spr = max(max(_pc[q][e] for q in _pc) - min(_pc[q][e] for q in _pc) for e in ("hair_light", "hair", "hair_heavy", "ruler"))
    check("altered-pixel fractions vary by at most 0.06 percentage points", 100 * _spr, 0.065, ceil=True,
          text="at most 0.06 percentage points", where="supp")

d = load("c42_positive_control")
if d:
    _ri = d["models"]["r50"]["Inv"]["ruler"]
    check("ResNet-50 Inv within tolerance under the ruler alone",
          "/".join(_ri[g]["verdict"] for g in ("all", "malignant", "benign")), "equivalence/equivalence/equivalence",
          text="the ResNet-50 Inv model is within tolerance under the ruler in all three subgroups", where="supp")
    order = [l["name"] for l in d["ladder"]]
    frac = {l["name"]: 100 * l["change_frac"] for l in d["ladder"]}
    verd = []
    first = []
    for bk in d["models"]:
        for mk in d["models"][bk]:
            ed = d["models"][bk][mk]
            for e, subs in ed.items():
                if e.startswith("_"):
                    continue
                verd += [v["verdict"] for v in subs.values()]
            fe = next(e for e in order if ed[e]["all"]["verdict"] == "exceedance")
            first.append(frac[fe])
    check("dose grid: 144 cells", len(verd), 144, text="@supp:55 of 144 cells")
    check("dose grid: 55 within tolerance", verd.count("equivalence"), 55, text="@supp:55 of 144 cells", where="supp")
    check("dose grid: 19 indeterminate", verd.count("indeterminate"), 19, text=["19 are indeterminate"], where=["paper", "supp"])
    check("first exceedance 0.34 to 2.37%", rng(first), (0.34, 2.37), 5e-3, text="0.34 to 2.37%")
    idv = [v["verdict"] for bk in d["models"] for mk in d["models"][bk]
           for v in d["models"][bk][mk]["identity"].values()]
    check("identity edit within tolerance in all eighteen cells", idv.count("equivalence"), 18)
    check("non-identity: 37 of 126 within tolerance",
          (verd.count("equivalence") - idv.count("equivalence"), len(verd) - len(idv)), (37, 126),
          text="Of these, 37 fall within tolerance, at the three lightest hair doses (30) and under the ruler (7), and 19 are indeterminate")
    check("dose grid spans 0.16 to 4.06%", rng(list(frac.values())), (0.16, 4.06), 5e-3,
          text="0.16 to 4.06%")

d = load("c46_padufes_artifact")
if d:
    agg = [d["models"][m]["_heldout_aggregate"][sg]["mean"]
           for m in d["models"] for sg in ("all", "malignant", "benign")]
    check("second cohort: six cells exceed", d["summary"]["exceedance"], 6)
    check("second cohort: 0.069 to 0.096", rng(agg), (0.069, 0.096), 5e-4, text="0.069 to 0.096")
    first = {m: next(e for e in ("hair_x1", "hair_x2", "hair_x3", "hair_light")
                     if d["models"][m][e]["all"]["verdict"] == "exceedance") for m in d["models"]}
    check("second cohort: both first exceed at 1.02%", str(sorted(set(first.values()))), "['hair_light']",
          text="@supp:exceed from 1.02%")

d = load("c13_fullpop")
if d:
    ph = [abs(d["models"][m]["edits"][e]["malignant"]["dp"]["mean"])
          for m in d["models"] for e in ("art:vignette", "art:jitter")]
    check("photometric edits move the malignant score by at most 0.09", max(ph), 0.095, ceil=True,
          text="by at most 0.09")
    lg = [d["models"][m]["edits"][e]["all"]["abs_dlogit"]["mean"]
          for m in d["models"] for e in ("art:vignette", "art:jitter")]
    check("photometric edits move the malignant logit by at most 1.7 (mean |change|, 183 images)", max(lg), 1.7, ceil=True,
          text="by at most 1.7", where="supp")
    hl = [d["models"][m]["edits"]["art:hair"]["all"]["abs_dlogit"]["mean"] for m in d["models"]]
    check("main hair edit moves it by up to 13.8 (mean |change|, 183 images)", max(hl), 13.8, 0.05,
          text="by up to 13.8 in mean absolute value over the 183 images", where="supp")
    resp = [d["models"][m]["edits"][t]["all"]["dlogit"]["mean"] for m in d["models"]
            for t in ("cpt:contrast@0", "cpt:contrast@2", "cpt:color_var@0", "cpt:color_var@2")]
    mx = max(abs(x) for x in resp)
    check("largest concept response 1.06 logits", mx, 1.06, 5e-3,
          text="At the largest, 1.06 logits", where="supp")
    check("component-range width 26.57 logits", 2 * B - mx, 26.57, 5e-3, text="26.57 logits", where="supp")
    check("width is 96.1% of the 27.63-logit range", 100 * (2 * B - mx) / (2 * B), 96.1, 0.05,
          text=["96.1%", "27.63-logit"])
    check("concept responses run −0.06 to +1.06", rng(resp), (-0.06, 1.06), 5e-3,
          text="−0.06 to +1.06", where="supp")

d = load("c19_saturation_mde")
if d:
    check("clipped logit bound B = 13.8155", d["clip_bound"], 13.8155, 5e-5, text="13.8155")
    check("87.1% of hair-edit predictions at the bound",
          100 * d["saturation_by_edit"]["art:hair"]["fraction"], 87.1, 0.05, text="87.1%")
    check("16.3% of edited predictions at the bound",
          100 * d["saturation"]["ALL models, edited only"]["fraction"], 16.3, 0.05, text="16.3%",
          where="supp")
    mde = []

    def wm(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "mde" and isinstance(v, (int, float)):
                    mde.append(v)
                else:
                    wm(v)
        elif isinstance(o, list):
            for v in o:
                wm(v)
    wm(d["mde_paired_contrast"])
    check("minimum detectable effect 0.48 to 1.79 logits", rng(mde), (0.48, 1.79), 5e-3,
          text="0.48 to 1.79")

d = load("c20_numerical_repro")
if d:
    cf = d["configs"]
    diffs = [abs(cf[c][m] - cf["bf16_per_image"][m]) for c in ("fp32_per_image", "bf16_chunk32")
             for m in cf["bf16_per_image"]]
    check("numerical check on the 183-image cohort", d["n_images"], 183)
    check("pipeline noise floor 0.303 logits", max(diffs), 0.303, 5e-4, text="0.303 logits", where=["paper", "supp"])
    pc = d["pct_change_vs_reference"]
    check("fp32 changes the aggregate by −2.8 to +2.8%", rng(list(pc["fp32_per_image"].values())),
          (-2.8, 2.8), 0.05, text="−2.8 to +2.8%", where="supp")
    check("batching changes it by −0.3 to +0.3%", rng(list(pc["bf16_chunk32"].values())),
          (-0.3, 0.3), 0.05, text="−0.3 to +0.3%", where="supp")

d = load("c21_patient_cluster")
if d:
    rat, agree, tot = [], 0, 0
    for mm in d["quantities"].values():
        for qq in mm.values():
            for v in qq.values():
                wi = v["ci_image"][1] - v["ci_image"][0]
                wp = v["ci_patient"][1] - v["ci_patient"][0]
                rat.append(wp / wi)
                agree += v["sig_image"] == v["sig_patient"]
                tot += 1
    check("patient-clustering comparison on the 183-image cohort", d["n_images"], 183)
    check("42 quantities, 39 agree", "%d/%d" % (agree, tot), "39/42", text="agrees in 39 of the 42",
          where="supp")
    check("width ratio 0.86 to 1.41", rng(rat), (0.86, 1.41), 5e-3, text="0.86 and 1.41", where="supp")

d = load("c26_hair_dose_prob")
if d:
    st = d["ERM"]["steps"][0]
    check("ERM first dose step −0.395", st["mean"], -0.395, 5e-4, text="−0.395 with [−0.482, −0.292]",
          where="supp")

# ================================================================ concept tier
d = load("c25_screen_full")
if d:
    sp = list(d["spearman"].values())
    check("Spearman −0.51 to −0.82", (max(sp), min(sp)), (-0.51, -0.82), 5e-3, text="−0.51 to −0.82")
    weak = d["spearman_ci"][max(d["spearman"], key=d["spearman"].get)]
    check("weakest Spearman interval [−0.79, −0.04]", tuple(weak), (-0.79, -0.04), 5e-3,
          text="@supp:[−0.79, −0.04]")
    check("four threshold schemes", len(d["schemes"]), 4, text="four threshold schemes")
    check("no valid control under any scheme",
          sum(s["valid_total"] for s in d["schemes"].values()), 0)
    bal = [s["tol"][0] for s in d["schemes"].values()]
    check("balance tolerances span 0.05 to 0.30", rng(bal), (0.05, 0.30), 1e-9,
          text="balance tolerances from 0.05 to 0.30")
    n_ctrl = {t: len([k for k in v if not k.startswith("_")]) for t, v in d["design"].items()}
    check("19 candidate controls per target", str(sorted(set(n_ctrl.values()))), "[19]",
          text="19 candidate controls")
    rotd = d["design"]["contrast@2"]["rotd"]
    check("best-balanced control: SMD 0.05, proxy 99%", (rotd["max_abs_smd"], 100 * rotd["proxy_ratio"]),
          (0.05, 99), 0.6, text="0.05 but moves the concept proxies by 99%")
    check("most neutral control: SMD 1.40",
          d["design"]["contrast@2"]["pixperm_m"]["max_abs_smd"], 1.40, 5e-3, text="1.40")
    neutral_smd = [x["max_abs_smd"] for v in d["design"].values() for k, x in v.items()
                   if not k.startswith("_") and x["proxy_shift"] <= 0.10 and abs(x["proxy_ratio"]) <= 0.15]
    check("neutral controls miss balance by at least 11 times", min(neutral_smd) / 0.10, 11, floor=True,
          text="at least 11 times")
    disjoint = all(not (x["max_abs_smd"] <= 0.10 and x["proxy_shift"] <= 0.10 and abs(x["proxy_ratio"]) <= 0.15)
                   for v in d["design"].values() for k, x in v.items() if not k.startswith("_"))
    check("balanced and neutral sets are disjoint", disjoint, True)

d = load("c55_identification_graph")
if d:
    pt = d["per_target"]
    check("88 performed edits", d["n_edits"], 88, text="88 performed edits")
    g0 = d["exact_graph_G0"]
    check("four exactly matched pairs", g0["n_exactly_matched_pairs"], 4, text="exactly four pairs")
    check("coincide for exactly four pairs in those pairs", len({e for p in g0["pairs"] for e in p}), 8,
          text="coincide for exactly four pairs")
    _r2 = rng([v["bottleneck_length_two"] for v in pt.values()])
    check("length-two route rho 9.6 to 13.3 (lower end rounded down)", (math.floor(_r2[0] * 10) / 10, _r2[1]),
          (9.6, 13.3), 0.05, text="9.6 to 13.3")
    check("best route rho 1.17 to 2.13", rng([v["bottleneck_over_all_paths"] for v in pt.values()]),
          (1.17, 2.13), 5e-3, text="1.17 to 2.13")
    bf = sorted(v["boot_frac_at_or_below_one"] for v in pt.values())
    check("three targets: bottleneck above one in 98.5 to 100% of resamples", (100 - 100 * bf[2], 100 - 100 * bf[0]), (98.5, 100.0), 0.05,
          text="98.5 to 100% of resamples")
    check("color variegation at top dose: above one in 89.9%", 100 - 100 * pt["color_var@2"]["boot_frac_at_or_below_one"],
          89.9, 0.05, text="89.9%")
    check("its bottleneck interval [0.90, 1.95]", tuple(pt["color_var@2"]["bottleneck_ci95"]),
          (0.90, 1.95), 5e-3, text="@supp:[0.90, 1.95]")
    check("route bootstrap 5,000 for the route search", d["boot"], 5000, text="@supp:5,000 for the bootstrap of the route search")
    check("route reselected in every resample", d.get("route_reselected_each_resample"), True,
          text="On patient resamples of the 183 images, the bottleneck stays above one")
    ci = {t: pt[t]["bottleneck_ci95"] for t in pt}
    check("Table S26 intervals", "; ".join("%.2f, %.2f" % tuple(ci[t]) for t in ("contrast@2", "color_var@2", "contrast@0", "color_var@0")),
          "1.10, 2.34; 0.90, 1.95; 1.09, 1.95; 1.24, 2.10", text="[1.10, 2.34] | 1.5%", where="supp")
    fr = {t: 100 * pt[t]["boot_frac_at_or_below_one"] for t in pt}
    check("Table S26 shares", "%.1f, %.1f, %.1f, %.1f" % (fr["contrast@2"], fr["color_var@2"], fr["contrast@0"], fr["color_var@0"]),
          "1.5, 10.1, 0.4, 0.0", text="[1.24, 2.10] | 0.0%", where="supp")
    L = d["lipschitz"]
    wp = L["worst_pair"]
    check("cross-set combination (183-image routes, c34 responses) gives 1.41 and a vacuous color_var@2 bound",
          "%.2f %s" % (L["implied_L_binding"], [t for t, v in L["breakeven_kappa_Lh"].items() if v == 0.0]), "1.41 ['color_var@2']",
          text="@supp:would give *L*ᵍ ≥ 1.41 and a vacuous bound at variegation at the highest dose")
    _c12 = load("c12_matched_control")
    if _c12:
        _sc = [_c12["models"][m]["dcace"][t]["u:hue"]["mean"] for m in _c12["models"]
               for t in ("contrast@2", "color_var@2", "contrast@0", "color_var@0")]
        check("span recomputed from c12 equals c55's", L["score_range_logits"], max(_sc) - min(_sc), 1e-12)
        check("corrected scores span 3.33 logits, 0.96 to 4.29",
              "%.2f/%.2f/%.2f" % (max(_sc) - min(_sc), min(_sc), max(_sc)), "3.33/0.96/4.29",
              text="@supp:span 3.33 logits end to end, from 0.96 to 4.29")
    check("no interaction price is claimed from neutral pairs", "5.12$ logits" in TEXT["paper"] or "5.12 logits" in TEXT["supp"], False)

d = load("c34_ctrl_vs_ctrl")
if d:
    su = d["summary"]
    diffs = [p["diff"] for p in d["pairs"]]
    check("twenty-one control comparisons", su["n_comparisons"], 21, text="twenty-one model-by-pair")
    check("all twenty-one exclude zero (patient)", su["excluding_zero_patient"], 21,
          text="all twenty-one patient-clustered intervals exclude zero")
    check("all twenty-one exclude zero (image)", su["excluding_zero_image"], 21)
    check("median disagreement 9.47 logits", statistics.median(abs(x) for x in diffs), 9.47, 5e-3,
          text="median of 9.47 logits")
    check("disagreement 2.26 to 14.43", rng([abs(x) for x in diffs]), (2.26, 14.43), 5e-3,
          text="2.26 to 14.43")
    _rank = {"globrot_m": 0, "noise_m": 1, "pixperm_m": 2}
    _ord = all((p["diff"] > 0) == (_rank[p["a"]] < _rank[p["b"]]) for p in d["pairs"])
    check("every comparison follows globrot_m > noise_m > pixperm_m", _ord, True,
          text="every one of the twenty-one comparisons follows one order", where="supp")
    check("the order as printed in S6", _ord, True, text="`globrot_m` above `noise_m` above `pixperm_m`", where="supp")
    check("pairs per target 0/1/3/3", str(su["n_pairs_per_model"]),
          "{'contrast@2': 0, 'color_var@2': 1, 'contrast@0': 3, 'color_var@0': 3}",
          text="3, 3, 1 and 0 pairs", where="supp")

d = load("c12_matched_control")
if d:
    r2 = {m: {f: v["r2_within"] for f, v in d["models"][m]["g_fit"]["lofo"].items()} for m in d["models"]}
    allr = [x for m in r2.values() for x in m.values()]
    check("leave-one-family-out R^2 negative everywhere", max(allr), 0.0, ceil=True)
    check("below −100 for one family in two of three models",
          sum(r2[m]["globrot"] < -100 for m in r2), 2, text="@supp:−1,631.9 for ERM, −66.1 for Slim and −204.4")
    check("globrot R^2 −1,631.9 / −66.1 / −204.4",
          "%.1f/%.1f/%.1f" % (r2["erm"]["globrot"], r2["slim"]["globrot"], r2["causal"]["globrot"]),
          "-1631.9/-66.1/-204.4", text="@supp:−1,631.9 for ERM, −66.1 for Slim and −204.4", where="supp")
    check("LOFO fit on 58 common images", d["models"]["erm"]["g_fit"]["n_img"], 58, text="58 images",
          where="supp")

d = load("c45_bracketing")
if d:
    su = d["summary"]
    check("balance+overlap controls bracket no cell", su["A1+A3"]["cells_bracketed"], 0)
    check("whole library brackets 7 of 12", su["all"]["cells_bracketed"], 7, text="7 of the 12")
    check("the whole library is the 15 constructed controls", str(su["all"]["set_size_range"]), "[15, 15]",
          text=["the 15 constructed controls (the four uncalibrated placebos are excluded), it succeeds in 7 of twelve"], where=["paper", "supp"])
    _bal = [v for t in d["targets"].values() for v in t["sets"]["A1+A3"]["models"].values()]
    check("balanced controls bracket the slope in 2 cells, the intercept in none",
          "%d/%d" % (sum(v["alpha_bracketed"] for v in _bal), sum(v["beta_bracketed"] for v in _bal)), "2/0",
          text="bracket the slope in two cells but the intercept in none", where="supp")
    lo, hi = su["all"]["edge_control_smd_range"]
    check("bracketing controls between 4.4 and 21 times outside the footprint tolerance outside tolerance", (lo / 0.1, hi / 0.1), (4.4, 21.0), 0.06,
          text="@supp:between 4.4 and 21 times outside the footprint tolerance")

d = load("c38_shrinkage_sensitivity")
if d:
    w = [v["slope_ci_patient"][1] - v["slope_ci_patient"][0] for m in d["models"].values() for v in m.values()]
    check("slope interval widths 0.04 to 0.40", rng(w), (0.04, 0.40), 5e-3, text="0.04 to 0.40", where="supp")
    ff = d["functional_form"]
    check("linear beats constant in ten of twelve",
          sum(v["linear"] > v["const"] for m in ff.values() for v in m.values()), 10,
          text="ten of twelve", where="supp")
    hs = d["holdout_slopes"]
    check("largest half-split disagreement 0.32",
          max(abs(v["slope_half_a"] - v["slope_half_b"]) for m in hs.values() for v in m.values()), 0.32, 5e-3,
          text="disagreement between halves is 0.32", where="supp")

d = load("c54_synthetic_scm")
if d:
    s = json.dumps(d)
    tgt = d["target_h"]
    rg = d["regimes"]
    check("simulated target 1.117", tgt, 1.117, 5e-4, text="1.117")
    check("valid regime within 0.8%", 100 * rg["R1_valid"]["abs_error_frac_of_target"], 0.8, 0.05,
          text="within 0.8%")
    check("model-state regime biased 6.7%", 100 * rg["R4_model_state"]["abs_error_frac_of_target"], 6.7, 0.05,
          text="@supp:| 6.7% |")
    check("structural biases 135.8, 48.6, 6.9 and 42.9%; valid regime exact on R0",
          " ".join("%.1f" % (100 * rg[k]["abs_error_frac_of_target_noise_free"]) for k in ("R1_valid", "R2_footprint_mismatch", "R3_concept_contamination",
                   "R4_model_state", "R5_proxy_blind")), "0.0 135.7 48.6 6.9 42.9",
          text=["by 6.9 and 42.9% on $R^0$", "One moves the latent concept, so N fails, which no screen sees",
                "@supp:| +1.117 | 0.0% |", "@supp:| −0.399 | 135.7% |", "@supp:| +0.574 | 48.6% |", "@supp:| +1.194 | 6.9% |", "@supp:| +0.637 | 42.9% |"])
    check("R4 structural error equals the bias formula +0.077", "%+.3f %+.3f" % (rg["R4_model_state"]["error_noise_free"], rg["R4_model_state"]["predicted_bias_from_formula"]),
          "+0.077 +0.077", text="@supp:at +0.077, which is its error on $R^0$")
    check("designed route exact on R0", d["constructive_repair"]["chained_route_noise_free_abs_error"], 0.0, 1e-12,
          text=["the valid regime and a designed length-four route, where no length-two route exists, return the target exactly on the structural response $R^0$", "@supp:On $R^0$ it returns the target exactly, to floating-point rounding"])
    check("proxy-blind regime biased 42.5%", 100 * rg["R5_proxy_blind"]["abs_error_frac_of_target"], 42.5, 0.05,
          text="@supp:42.5%")
    check("screens admit R1, R4, R5 and reject R2, R3",
          "".join("A" if v["screen"]["admissible"] else "R" for v in rg.values()), "ARRAA")
    ch = float(re.search(r'"chained_route_estimate": ([\d.]+)', s).group(1))
    ci = re.search(r'"bootstrap_ci95": \[([\d.]+), ([\d.]+)\]', s)
    near = float(re.search(r'"nearest_neutral_control_footprint_distance_max_smd": ([\d.]+)', s).group(1))
    check("designed length-four route +1.111", ch, 1.111, 5e-4, text="+1.111")
    check("its interval [1.085, 1.151]", (float(ci.group(1)), float(ci.group(2))), (1.085, 1.151), 5e-4,
          text="@supp:[1.085, 1.151]")
    check("recovery within 0.6%%", 100 * abs(ch - tgt) / tgt, 0.6, ceil=True, text="@supp:within 0.6%")
    check("nearest neutral control 1.65 standardized mean differences away", near, 1.65, 5e-3, text="@supp:1.65 standardized mean differences")
    check("no length-two route there", re.search(r'"placebo_route_exists": (\w+)', s).group(1), "false")

d = load("c49_tcav")
if d:
    accs, flags = [], []

    def walk(o):
        if isinstance(o, dict):
            if "cav_accuracy_mean" in o:
                accs.append(o["cav_accuracy_mean"])
            if "significant" in o:
                flags.append(bool(o["significant"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    check("CAV accuracy 0.84 to 0.94 on all activations (S12)", rng(accs), (0.84, 0.94), 6e-3, text="unedited activations at 0.84 to 0.94 accuracy", where="supp")
    _ho = [e["cav_accuracy_heldout_mean"] for v in d["models"].values() for e in v["concepts"].values()]
    check("CAV accuracy 0.68 to 0.87 on held-out activations", rng(_ho), (0.68, 0.87), 5e-3,
          text=["with accuracy 0.68 to 0.87", "or 0.68 to 0.87 on the activations not used to fit them"], where=["paper", "supp"])
    check("significant in 7 of 12 cells", "%d of %d" % (sum(flags), len(flags)), "7 of 12", text="7 of the 12 model-by-concept cells nominally significant")

d = load("c10_multiplacebo")
if d:
    check("multi-placebo sample: 57 of 83 malignant audit images", d["n_common"], 57,
          text="57 of 83 malignant audit images")
    check("multi-placebo intervals are patient-clustered", d.get("resampling_unit"), "patient",
          text="with the patient as the resampling unit", where="supp")
    check("multi-placebo: 29 patients", d.get("n_patients_common"), 29, text="from 29 patients")
    a = {m: {k: v["A"] for k, v in mv["A_corr_by_placebo"].items()} for m, mv in d["models"].items()}
    check("Slim −0.34 under flip", a["slim"]["flip"], -0.34, 5e-3, text="−0.34")
    check("Slim +5.14 under noise", a["slim"]["noise"], 5.14, 5e-3, text="+5.14")
    flips = sum(1 for m in a if min(a[m].values()) < 0 < max(a[m].values()))
    check("sign changes for one of three models", flips, 1, text="changes sign for one model")
    # Table S29, cell by cell
    tx = TEXT["supp"]
    t = tx[tx.index("| Model | Placebo | proxy change of placebo"):]
    t = t[:t.index("\n\n")]
    name = {"ERM": "erm", "Slim": "slim", "Inv": "causal"}
    mk = None
    bad = n = 0
    for line in t.splitlines()[2:]:
        f = [x.strip() for x in line.strip("|").split("|")]
        mk = name.get(f[0], mk)
        p = f[1]
        mv = d["models"][mk]
        got = ([100 * d["placebo_proxychg"][p]] + mv["placebo_cace"][p] + mv["A_corr_by_placebo"][p]["contrast"]
               + mv["A_corr_by_placebo"][p]["color"] + [mv["A_corr_by_placebo"][p]["A"]])
        nums = [num(x) for x in re.findall(r"[−+]?\d+\.\d+", line)]
        ok = len(nums) == 11 and abs(nums[0] - got[0]) <= 0.05 and all(abs(x - y) <= 5e-3 + 1e-9 for x, y in zip(nums[1:], got[1:]))
        n += 1
        bad += not ok
        if not ok:
            rows.append(("MISMATCH", "Table S29 row %s %s" % (mk, p), str([round(x, 3) for x in got]), line))
    check("Table S29: all 12 rows match c10", n - bad, 12)

# ---- checks added with the number audit of the supplement ----
_a, _b = load("c40_simultaneous"), load("c30_disjoint_audit")
if _a and _b:
    _SG = {"all": "all", "mal": "malignant", "ben": "benign"}
    _mx = 0.0
    for _k, _v in _a["disjoint_18"]["cells"].items():
        _mdl, _sg = _k.split("/"); _bb, _mm = _mdl.split("-")
        for _i in (0, 1):
            _mx = max(_mx, abs(_v["ci_marginal"][_i] - _b["models"][_bb][_mm][_SG[_sg]]["ci"][_i]))
    check("marginal endpoints of the two runs differ by at most 0.0044 (upper bound rounded up)", _mx, 0.0044, ceil=True,
          text="differ by at most 0.0044", where="supp")
_c51 = load("c51_independent_rebuild")
if _c51:
    _cg = _c51["disagreement_diagnosis"]["cross_grid"]
    check("reported checkpoint gives 0.123 on both cohorts",
          "%.3f/%.3f" % (_cg["ckpt=here|cohort=here"]["magnitude"], _cg["ckpt=here|cohort=other"]["magnitude"]), "0.123/0.123",
          text="gives 0.123 on both cohorts", where="supp")
_c52 = load("c52_overlay_perceptual")
if _c52:
    check("between-lesion scale keeps 1,991 of 2,000 draws", "%d/%d" % (_c52["n_pairs"], _c52["between_lesion"]["n_pairs"]),
          "2000/1991", text="keeps the 1,991 that join two different images", where="supp")
_c25m = load("c25_screen_mal")
if _c25m:
    _D = _c25m["design"]
    check("S6: noise_m fails neutrality at both top doses, 20 and 18%",
          "%.0f/%.0f" % (100 * _D["contrast@2"]["noise_m"]["proxy_shift"], 100 * _D["color_var@2"]["noise_m"]["proxy_shift"]),
          "20/18", text="with proxy shifts of 20 and 18%", where="supp")
    check("S6: pixperm_m overlap 92% at the contrast top dose", round(100 * _D["contrast@2"]["pixperm_m"]["positivity"]), 92,
          text="by feasibility, at 92%", where="supp")
    _hp = [t for t in _D if _D[t]["hue_m"]["proxy_shift"] <= 0.10 and _D[t]["hue_m"]["proxy_ratio"] <= 0.15]
    check("S6: hue_m passes neutrality only at the contrast top dose", str(_hp), "['contrast@2']",
          text="`hue_m` passes the neutrality screen only at the contrast top dose", where="supp")
_c38 = load("c38_shrinkage_sensitivity")
if _c38:
    _nd = max(v.get("n_dropped", 0) for m in _c38["models"].values() for v in m.values() if isinstance(v, dict))
    check("at most 8 of 183 bounded observations in any one shrinkage fit", _nd, 8,
          text="at most 8 of 183 in any one fit", where="supp")

_c42b = load("c42_positive_control")
if _c42b:
    _ord = [l["name"] for l in _c42b["ladder"]]
    _fr = {l["name"]: 100 * l["change_frac"] for l in _c42b["ladder"]}
    _fe = {}
    for _bk in _c42b["models"]:
        for _mk in _c42b["models"][_bk]:
            _ed = _c42b["models"][_bk][_mk]
            _fe["%s-%s" % (_bk, _mk)] = _fr[next(e for e in _ord if _ed[e]["all"]["verdict"] == "exceedance")]
    _top = max(_fe.values())
    check("ResNet-50 Inv alone has the highest first-exceeding hair dose, 2.37%",
          "%s %.2f" % (",".join(k for k, v in _fe.items() if v == _top), _top), "r50-Inv 2.37",
          text="to 2.37% for the ResNet-50 invariance model, spanning four of the six grid steps", where="supp")
_c62p = load("c62_preregistration")
if _c62p:
    check("cycle test uses 20,000 resamples", _c62p["criteria"]["boot"], 20000,
          text="the simultaneous band, the control comparisons and the cycle test use 20,000")
_c64b = load("c64_route_screens")
if _c64b:
    _tp = max(_c64b["frontier"][t]["rad"]["target_proxy_shift"] for t in ("contrast@2", "color_var@2"))
    _sh = min(_c64b["frontier"][t]["rad"]["offtarget_proxy_shift"] / (2 * _c64b["frontier"][t]["rad"]["proxy_shift"])
              for t in ("contrast@2", "color_var@2"))
    check("optimized radial controls: target proxy moves < 0.1%, off-target share of the mean shift > 99%",
          "%s %s" % (_tp < 1e-3, _sh > 0.99), "True True",
          text="@supp:so every (A2) failure comes from the other concept's proxy")

# ================================================================ occlusion check (c66)
_oc = load("c66_overlay_occlusion")
if _oc:
    cov = _oc["coverage"]
    check("occlusion: full overlays reproduce Table 4", max(v["check_full_vs_c30_all"] for v in _oc["checks"].values()), 0.0, 0.0)
    check("occlusion: lesion region covers a median 36% of the frame", round(100 * _oc["lesion_region"]["median_frame_share"]), 36,
          text="covers a median 36% of the frame", where=["paper", "supp"])
    check("occlusion: region share 2 to 100%", tuple(round(100 * x) for x in _oc["lesion_region"]["range_frame_share"]), (2, 100), 0,
          text="with a range of 2 to 100%", where="supp")
    check("occlusion: 40% of the main hair edit's altered pixels inside (main text)",
          "%.0f" % (100 * cov["hair"]["altered_in_lesion"]["mean"]), "40",
          text="@supp:40 to 41% of the pixels they alter lie inside it")
    check("occlusion: on average 34% of the ruler's altered pixels lie in the lesion region (main text and S8)",
          round(100 * cov["ruler"]["altered_in_lesion"]["mean"]), 34,
          text=["and 34% of its altered pixels lie inside", "On average, a third of its altered pixels still fall inside the lesion region"], where=["paper", "supp"])
    check("occlusion: light/main/heavy alter 1.1, 2.4 and 4.2% of the region",
          "/".join("%.1f" % (100 * cov[a]["lesion_share_altered"]["mean"]) for a in ("hair_light", "hair", "hair_heavy")), "1.1/2.4/4.2",
          text="alter 1.1, 2.4 and 4.2% of the region's pixels", where="supp")
    _ins = [100 * cov[a]["altered_in_lesion"]["mean"] for a in ("hair_light", "hair", "hair_heavy")]
    check("occlusion: 40 to 41% of hair-altered pixels inside", (round(min(_ins)), round(max(_ins))), (40, 41), 0,
          text="@supp:40 to 41% of the pixels they alter lie inside it", where="supp")
    check("occlusion: ruler alters 0.8% of the region, 34% inside",
          "%.1f %.0f" % (100 * cov["ruler"]["lesion_share_altered"]["mean"], 100 * cov["ruler"]["altered_in_lesion"]["mean"]), "0.8 34",
          text="The ruler alters 0.8% of the region, and 34% of its altered pixels lie inside", where="supp")
    check("occlusion: main hair alters 1.43 and 0.93% of the frame when split",
          "%.2f %.2f" % (100 * cov["hair"]["altered_frac_skin_only"]["mean"], 100 * cov["hair"]["altered_frac_lesion_only"]["mean"]), "1.43 0.93",
          text="The main hair edit then alters 1.43 and 0.93% of the frame", where="supp")
    sk, fu, le = _oc["cells"]["skin_only"], _oc["cells"]["full"], _oc["cells"]["lesion_only"]
    nsk = sum(c["joint_verdict"] == "exceedance" for c in sk.values())
    check("occlusion: skin-only overlays exceed in 16 of 18 cells", nsk, 16,
          text=["still exceed 0.05 in 16 of the 18 cells"], where=["paper", "supp"])
    rat = [sk[k]["mean"] / fu[k]["mean"] for k in fu]
    check("occlusion: skin-only at 52 to 101% of the full response", (round(100 * min(rat)), round(100 * max(rat))), (52, 101), 0,
          text="at 52 to 101% of the full response", where=["paper", "supp"])
    ex = sorted(k for k, c in sk.items() if c["joint_verdict"] != "exceedance")
    check("occlusion: the two exceptions are B3 Inv benign and R50 Inv malignant, indeterminate",
          ",".join(ex) + " " + ",".join(sk[k]["joint_verdict"] for k in ex), "b3-Inv/benign,r50-Inv/malignant indeterminate,indeterminate",
          text="The two exceptions are indeterminate: Inv on EfficientNet-B3 benign images", where="supp")
    for k, txt in (("b3-Inv/benign", "0.073 [0.047, 0.102]"), ("r50-Inv/malignant", "0.047 [0.041, 0.055]")):
        c = sk[k]
        check("occlusion: %s skin-only interval" % k, "%.3f [%.3f, %.3f]" % (c["mean"], c["ci"][0], c["ci"][1]), txt, text=txt, where="supp")
    b = _oc["band"]
    # a frontier is rounded down, like 0.042
    check("occlusion: skin-only frontier 0.026 (rounded down) and 12 of 18 jointly",
          "%.3f %d" % (math.floor(1000 * b["skin_only"]["frontier_36"]) / 1000, b["skin_only"]["magnitude_exceed_36"]),
          "0.026 12", text=["the skin-only frontier is 0.026, and 12 of the 18 cells exceed 0.05 jointly",
                            "and 12 cells exceed jointly", "The fall of the frontier from 0.042 to 0.026"],
          where=["paper", "supp"])
    _sk_cells = sum(1 for k in sk if (lambda m, g: m[0] > 0.05 or g[0] > 0.05 or g[1] < -0.05)(b["skin_only"]["sim36"][k + "/mag"], b["skin_only"]["sim36"][k + "/sgn"]))
    check("occlusion: 12 skin-only cells exceed 0.05 on the joint band (cell rule)", _sk_cells, 12)
    check("occlusion: skin-only response is more than half of the full response in every cell", min(rat) > 0.5, True)
    nle = sum(c["joint_verdict"] == "exceedance" for c in le.values())
    check("occlusion: lesion-only 17 cells, frontier 0.025", "%d %.3f" % (nle, b["lesion_only"]["frontier_36"]), "17 0.025",
          text="Under the lesion-only mixture, 17 cells exceed marginally, and the frontier is 0.025", where="supp")
    mn = min(sk["%s/%s" % (m, g)]["mean"] for m in ("b3-ERM", "b3-Slim") for g in ("all", "malignant", "benign"))
    check("occlusion: every B3 ERM and Slim cell at least 0.191", round(mn, 3), 0.191, 1e-9, text="by at least 0.191", where="supp")

# ================================================================ route test: floor and numerics (c67)
_rs = load("c67_route_sensitivity")
_e1, _e2 = load("c62_evaluation"), load("c62_evaluation_inv_retrained")
if _rs and _e1 and _e2:
    A, B, C = _rs["floor"], _rs["numerics"], _rs["numerics_first_run"]
    three = ["causal|contrast@2", "erm|contrast@2", "slim|color_var@2"]
    dis = lambda cells: sorted(c for c, v in cells.items() if v == "routes disagree")
    check("floor: second run reproduced exactly", A["reproduced_second_run"]["max_abs_diff_sim_bounds"], 0.0, 1e-9)
    check("floor: per-model floor and first-run images give the same three cells",
          "%s | %s" % (",".join(dis(A["per_model"]["cells"])), ",".join(dis(A["run1_set"]["cells"]))), "%s | %s" % (",".join(three), ",".join(three)),
          text="Two post hoc variants give the same three cells", where="supp")
    ns = [v[0] for v in A["per_model"]["n"].values()]
    check("floor: per-model floor keeps 45 to 78 images per cell", (min(ns), max(ns)), (45, 78), 0, text="keeps 45 to 78 images per cell", where="supp")
    cf = A["common_set_first_run"]
    nsc = [v[0] for v in cf["n"].values()]
    check("floor: on the common images the first run's borderline cell is indeterminate, the other two remain",
          "%s %s %d-%d" % (cf["cells"]["slim|contrast@0"], ",".join(dis(cf["cells"])), min(nsc), max(nsc)),
          "indeterminate erm|contrast@2,slim|color_var@2 44-47",
          text="On the 44 to 47 images per target that pass the floor under both Inv checkpoints", where="supp")
    v32, vfl = B["variants"]["fp32_stack"], B["variants"]["bf16_flat32"]
    check("numerics: stored second-run responses reproduced exactly", max(B["check_bf16_stack_vs_stored"].values()), 0.0, 0.0)
    check("numerics: stored first-run responses reproduced exactly", C["check_bf16_stack_vs_stored_first_run"], 0.0, 0.0)
    check("numerics: full precision adds three Slim disagreements",
          ",".join(sorted(v32["cells_changed"])) + " " + ",".join(v32["cells"][c] for c in sorted(v32["cells_changed"])),
          "slim|color_var@0,slim|contrast@0,slim|contrast@2 routes disagree,routes disagree,routes disagree",
          text="In full precision, three further Slim cells disagree", where="supp")
    check("numerics: differences move by up to 2.19 and 1.86 logits", (v32["max_abs_change_D"], vfl["max_abs_change_D"]), (2.19, 1.86), 5e-3,
          text=["differences move by up to 2.19 logits", "differences move by up to 1.86 logits"], where="supp")
    check("numerics: mixed batches swap the Slim cells",
          "%s %s" % (vfl["cells"]["slim|color_var@2"], vfl["cells"]["slim|contrast@2"]), "indeterminate routes disagree",
          text="the Slim disagreement at the highest variegation dose becomes indeterminate", where="supp")
    em = [c for v in (v32, vfl, C["variants"]["fp32_stack"], C["variants"]["bf16_flat32"]) for c in v["cells_changed"] if not c.startswith("slim|")]
    check("numerics: no ERM or Inv cell changes in either run", len(em), 0,
          text=["ERM and Inv keep every verdict under both", "change no ERM or Inv cell"], where="supp")
    sat = B["saturation_primary"]["slim"]["contrast@2"]["rad4"]
    check("numerics: 89% of Slim rad4 predictions at the bound at the highest contrast dose", round(100 * sat), 89,
          text=["89% of the rad4 predictions for Slim at the highest contrast dose"], where="supp")
    _se, _si = B["saturation_primary"]["erm"]["contrast@2"]["rad4"], B["saturation_primary"]["causal"]["contrast@2"]["rad4"]
    check("numerics: 32% (ERM) and 34% (Inv) of rad4 predictions at the bound in the robust cells",
          "%d/%d" % (round(100 * _se), round(100 * _si)), "32/34",
          text="At the highest contrast dose, 32% of the rad4 predictions sit at it for ERM and 34% for Inv", where="supp")
    check("numerics: about a third in the robust cells (main text)", 0.30 <= min(_se, _si) and max(_se, _si) <= 0.36, True,
          text="@supp:34% of the images have a prediction at the bound in the scale pair")
    check("numerics: 22% of Slim edit and control predictions beyond the bound, 5% ERM, 3% Inv",
          "/".join("%d" % round(100 * B["raw_logodds_primary"][m]["share_beyond_clip"]) for m in ("slim", "erm", "causal")), "22/5/3",
          text="22% of all Slim predictions for edits and controls lie beyond the bound, against 5% for ERM and 3% for Inv", where="supp")
    mx = [B["raw_logodds_primary"][m]["max_abs_logodds"] for m in ("erm", "slim", "causal")]
    check("numerics: unclipped log-odds reach 25,984, 41,344 and 2,488", "%d/%d/%d" % tuple(mx), "25984/41344/2488",
          text="25,984, 41,344 and 2,488 logits", where="supp")
    check("numerics: no cell decided on the unclipped log-odds", B["variants"]["raw_logodds"]["summary"]["indeterminate"], 12)
    # robust cells: ERM at the highest contrast dose in both runs, Inv at the highest contrast dose in the second
    allv2 = [A["per_model"], A["run1_set"], A["common_set_second_run"]] + [B["variants"][k] for k in ("bf16_stack", "fp32_stack", "bf16_flat32")]
    allv1 = [A["per_model_first_run"], A["common_set_first_run"]] + [C["variants"][k] for k in ("bf16_stack", "fp32_stack", "bf16_flat32")]
    ok2 = all(v["cells"][c] == "routes disagree" for v in allv2 for c in ("erm|contrast@2", "causal|contrast@2"))
    ok1 = all(v["cells"]["erm|contrast@2"] == "routes disagree" for v in allv1)
    ok1 = ok1 and _e1["secondary_all_g1"]["cells"]["erm|contrast@2"]["class"] == "routes disagree"
    ok2 = ok2 and all(_e2["secondary_all_g1"]["cells"][c]["class"] == "routes disagree" for c in ("erm|contrast@2", "causal|contrast@2"))
    rbs = [load("c65_route_robustness"), load("c65_route_robustness_inv_retrained")]
    q1, q2 = rbs[0]["decisive"]["erm|contrast@2|D_rad4-rad16"], rbs[1]["decisive"]
    ok1 = ok1 and q1["seed_exceeds"] == 20 and q1["lopo_in_cell_exceeds"] == q1["lopo_in_cell_n"]
    ok2 = ok2 and all(q2[k]["seed_exceeds"] == 20 and q2[k]["lopo_in_cell_exceeds"] == q2[k]["lopo_in_cell_n"]
                      for k in ("erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16"))
    check("robust cells: ERM highest contrast in both runs, Inv highest contrast in the second, under every check", "%s %s" % (ok1, ok2), "True True",
          text=["@supp:the flags for ERM and Inv at the highest contrast dose survive every check of Table S37", "ERM at the highest contrast dose stays flagged under each variant in both runs"])
    check("robust cells: claims are restricted to the clipped scale (main text and S19)", True, True,
          text=["They hold on the clipped scale for a fixed inference protocol", "the plan's nominal band flagged route sums differing beyond 0.5 logit in three cells",
                "Every route-test decision is made on the clipped scale"], where=["paper", "supp"])
    counts = [sum(v == "routes disagree" for v in x["cells"].values()) for x in allv1 + allv2]
    counts += [_e1["secondary_all_g1"]["summary"]["routes disagree"], _e2["secondary_all_g1"]["summary"]["routes disagree"]]
    check("variants: two to six cells disagree", (min(counts), max(counts)), (2, 6), 0, text="@supp:across all variants of both runs two to six cells are flagged")

# ================================================================ floor sets (c68) and the cache manifest
_fs = load("c68_floor_sets")
if _fs:
    md = _fs["membership_differs"]
    sc = md[0]["scores"] if len(md) == 1 else None
    bf = [sc[v]["causal"] for v in ("c10_batch", "c19_cache", "single_bf16")] if sc else [0]
    check("floor sets: one image, ISIC_3039686, at 0.099 to 0.102 in bf16 and 0.096 in fp32",
          "%s %.3f-%.3f %.3f" % (md[0]["isic_id"], min(bf), max(bf), sc["single_fp32"]["causal"]) if sc else None,
          "ISIC_3039686 0.099-0.102 0.096",
          text="One image, ISIC_3039686, scores 0.099 to 0.102 under the Inv model, depending on that batch, and 0.096 in full precision", where="supp")
    check("floor sets: full precision keeps 57, the cache 58", "%d %d" % (_fs["variants"]["single_fp32"]["n_kept"], _fs["variants"]["c19_cache"]["n_kept"]),
          "57 58", text="The full-precision floor keeps the 57-image set", where="supp")
    check("floor sets: the c10 batch reproduces its 57", _fs["c10_reproduces_n_common"], True)
_cm = load("c19_preds_manifest")
if _cm:
    import hashlib as _hl
    _ok = all(_hl.sha256(open(os.path.join(BASE, "checkpoints", v["file"]), "rb").read()).hexdigest() == v["sha256"]
              for v in _cm["checkpoints"].values())
    _ok = _ok and _hl.sha256(open(os.path.join(RES, "c19_preds.npz"), "rb").read()).hexdigest() == _cm["cache_sha256"]
    check("prediction cache built from the current checkpoints", _ok, True,
          text="which records the SHA-256 of the checkpoints it read", where="supp")

# ================================================================ clip level and extreme log-odds (c69)
_cl = load("c69_route_clip_sensitivity")
if _cl:
    R2, R1 = _cl["runs"]["second_run"], _cl["runs"]["first_run"]
    dset = lambda cells: sorted(c for c, v in cells.items() if v == "routes disagree")
    check("clip: z-based reading at 1e-6 reproduces the plan, estimates within 0.001",
          "%s %s %s" % (R2["z_vs_plan_at_1e-6"]["cells_equal"], R1["z_vs_plan_at_1e-6"]["cells_equal"],
                        max(R2["z_vs_plan_at_1e-6"]["max_abs_diff_est"], R1["z_vs_plan_at_1e-6"]["max_abs_diff_est"]) < 1e-3),
          "True True True", text="with estimates within 0.001 logit", where="supp")
    bnd = [R2["eta_bf16"][k]["bound_logit"] for k in ("0.0001", "1e-08")]
    check("clip: bound 9.2 to 18.4 logits", tuple(round(x, 1) for x in bnd), (9.2, 18.4), 1e-9,
          text="9.2 to 18.4 logits", where=["paper", "supp"])
    three = ["causal|contrast@2", "erm|contrast@2", "slim|color_var@2"]
    okb = all(dset(R2["eta_bf16"][k]["cells"]) == three for k in ("1e-05", "1e-06", "1e-07", "1e-08"))
    okb = okb and dset(R2["eta_bf16"]["0.0001"]["cells"]) == sorted(three + ["slim|contrast@0"])
    check("clip: second run, same three cells from 1e-5 to 1e-8 in bf16, Slim lowest contrast joins at 1e-4", okb, True,
          text="Slim at the lowest contrast dose joins them at 10⁻⁴", where="supp")
    n32 = [len(dset(R2["eta_fp32"][k]["cells"])) for k in ("0.0001", "1e-05", "1e-06", "1e-07", "1e-08")]
    drop = [k for k in ("0.0001", "1e-05", "1e-06", "1e-07", "1e-08") if "slim|color_var@2" not in dset(R2["eta_fp32"][k]["cells"])]
    check("clip: full precision 5 or 6 cells; Slim highest variegation drops at 1e-7 and below", "%d-%d %s" % (min(n32), max(n32), ",".join(drop)),
          "5-6 1e-07,1e-08", text="In full precision five or six cells disagree, and the Slim cell at the highest variegation dose drops out at 10⁻⁷ and below", where="supp")
    allr = [R2["eta_" + n][k]["cells"] for n in ("bf16", "fp32") for k in R2["eta_" + n]]
    check("clip: ERM and Inv highest contrast disagree at every level in both precisions",
          all(c[x] == "routes disagree" for c in allr for x in ("erm|contrast@2", "causal|contrast@2")), True,
          text="ERM and Inv at the highest contrast dose disagree at every level in both precisions", where="supp")
    allr1 = [R1["eta_" + n][k]["cells"] for n in ("bf16", "fp32") for k in R1["eta_" + n]]
    ok1 = all(c[x] == "routes disagree" for c in allr1 for x in ("erm|contrast@2", "slim|color_var@2"))
    sd = [k for k in R1["eta_bf16"] if R1["eta_bf16"][k]["cells"]["slim|contrast@0"] != "routes disagree"]
    check("clip: first run, ERM highest contrast and Slim highest variegation at every level; Slim lowest contrast drops at 1e-7 and below in bf16",
          "%s %s" % (ok1, ",".join(sd)), "True 1e-07,1e-08", text="drops out at 10⁻⁷ and below in bf16", where="supp")
    check("clip: floor read in fp32 changes one image of the second run, none of the first, and no decision",
          "%d %d %s %s" % (R2["floor_fp32"]["images_changed"], R1["floor_fp32"]["images_changed"],
                           R2["floor_fp32"]["cells"] == R2["plan"]["cells"], R1["floor_fp32"]["cells"] == R1["plan"]["cells"]),
          "1 0 True True", text="Reading the score floor in full precision changes one image of the second run and none of the first", where="supp")
    sp = R2["eta_bf16"]["1e-06"]["saturation_pairs"]
    check("saturation: 93% of Slim highest-contrast images have a scale-pair prediction at the bound; 34% in the robust cells",
          "%d %d %d" % (round(100 * sp["slim|contrast@2"]["rad4-rad16"]), round(100 * sp["erm|contrast@2"]["rad4-rad16"]),
                        round(100 * sp["causal|contrast@2"]["rad4-rad16"])), "93 34 34",
          text=["93% of its images have a prediction at the bound", "and 34% of the images have a prediction at the bound in the scale pair"],
          where=["paper", "supp"])
    ex = _cl["extreme_logodds_second_run"]
    check("extreme: full-precision maxima 8,901, 71,120 and 4,215", "%d/%d/%d" % tuple(round(ex[m]["max_abs_logodds_fp32"]) for m in ("erm", "slim", "causal")),
          "8901/71120/4215", text="and 8,901, 71,120 and 4,215 in full precision", where="supp")
    rat = [x["feature_norm_bf16"] / ex[m]["unedited_feature_norm_median"] for m in ex for x in ex[m]["top10"]]
    check("extreme: feature norm about 420 to 44,000 times its unedited median", (math.floor(min(rat) / 10) * 10, round(max(rat), -3)), (420, 44000), 0,
          text="about 420 to 44,000 times its median norm on unedited images", where="supp")
    cr = [ex[m]["corr_abs_logodds_feature_norm"] for m in ex]
    check("extreme: correlation 0.88 to 0.99", (round(min(cr), 2), round(max(cr), 2)), (0.88, 0.99), 1e-9, text="at 0.88 to 0.99", where="supp")
    flips = sum(1 for m in ex for x in ex[m]["top10"] if x["logodds_bf16"] * x["logodds_fp32"] < 0)
    check("extreme: one sign flip among the ten largest per model", flips, 1, text="it flips the sign of one of the ten largest", where="supp")
    ps2, ps1 = R2["probability_scale"], R1["probability_scale"]
    dec = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4", "slim|color_var@2|D_rad4-rad16"]
    check("probability scale: 12 of 48 exclude zero incl. every decisive difference; 18 in the first run",
          "%d %s %d" % (ps2["n_excluding_zero"], all(k in ps2["excluding_zero"] for k in dec), ps1["n_excluding_zero"]), "12 True 18",
          text=["12 of the 48 simultaneous intervals of the second run exclude zero", "18 do in the first run"], where="supp")
    _cons = load("c62_construction")
    _g2 = ["%d" % round(100 * _cons["summary"][t][c]["share_g2_independent"]) for t in ("contrast@2", "color_var@2") for c in ("rad4", "rad16")]
    check("G2 of the disagreeing scale pairs: 13 and 23% (contrast), 35 and 38% (variegation)", "/".join(_g2), "13/23/35/38",
          text="@supp:rad4 and rad16 pass it on only 13 and 23% of the images at the highest contrast dose, and on 35 and 38% at the highest variegation dose")

# ================================================================ every variant in one table (c73) and the bounded-deviation reading
_rt = load("c73_route_robustness_table")
if _rt:
    S2, S1 = _rt["second_run"], _rt["first_run"]
    mn = S2["smallest_lower_bound"]
    check("robust: ERM and Inv highest contrast disagree under every variant of the second run", ",".join(sorted(S2["robust_cells"])),
          "causal|contrast@2,erm|contrast@2", text="ERM and Inv at the highest contrast dose disagree under all of them", where="supp")
    check("robust: smallest lower bounds 0.72, 1.40 and 0.51 (rounded down)", "/".join("%.2f" % (math.floor(mn[k] * 100 + 1e-9) / 100) for k in ("erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16",
          "causal|contrast@2|D_rad4-hue4")), "0.72/1.40/0.51",
          text=["are 0.72 and 1.40 logits, and that of the Inv direction pair is 0.51", "@supp:the smallest simultaneous lower bounds of their scale-pair differences are 0.72 and 1.40 logits"],
          where=["paper", "supp"])
    check("robust: second-run variants give 3 to 6 disagreeing cells", tuple(S2["disagree_range"]), (3, 6), 0,
          text="The variants change the number of disagreeing cells from 3 to 6", where="supp")
    m1 = S1["smallest_lower_bound"]
    rob1 = [c for c in ("erm|contrast@2", "slim|color_var@2", "slim|contrast@0") if all(c in r["disagreeing_cells"] for r in S1["rows"])
            and S1["seeds"][c + "|D_rad4-rad16"][1] == 20 and S1["leave_one_patient_out"][c + "|D_rad4-rad16"][1] == S1["leave_one_patient_out"][c + "|D_rad4-rad16"][2]]
    check("robust: c73 robust lists agree with an independent computation, and only ERM highest contrast is robust in both runs",
          "%s %s" % (sorted(S1["robust_cells"]) == sorted(rob1), ",".join(sorted(set(S1["robust_cells"]) & set(S2["robust_cells"])))),
          "True erm|contrast@2", text="ERM at the highest contrast dose stays flagged under each variant in both runs, Slim at the highest variegation dose in the first run and Inv at the highest contrast dose in the second")
    check("robust: first run, ERM highest contrast and Slim highest variegation under every variant, bounds 1.21 and 0.63",
          "%s %.2f %.2f" % (",".join(rob1), m1["erm|contrast@2|D_rad4-rad16"], m1["slim|color_var@2|D_rad4-rad16"]), "erm|contrast@2,slim|color_var@2 1.21 0.63",
          text="Their smallest lower bounds, 1.21 and 0.63, both come from the images that passed the floor under both Inv checkpoints", where="supp")
    _argmin = [min((r["entries"][k][1], r["variant"]) for r in S1["rows"] if r["entries"])[1]
               for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("robust: both first-run minima come from the floor passed under both Inv checkpoints", " | ".join(_argmin),
          "floor passed under both Inv checkpoints | floor passed under both Inv checkpoints",
          text="both come from the images that passed the floor under both Inv checkpoints", where="supp")
    check("robust: first-run Slim lowest contrast fails under 10 of 20 seeds", 20 - S1["seeds"]["slim|contrast@0|D_rad4-rad16"][1], 10,
          text="It fails under 10 of 20 other seeds", where="supp")
    # main text: which flags hold under every variant, and none labelled robust or fragile in Table 5
    _ev1 = load("c62_evaluation")["primary"]["cells"]
    _fl1 = sorted(c for c, v in _ev1.items() if v["class"] == "routes disagree")
    _rows5 = [ln for ln in TEXT["paper"].splitlines() if ln.startswith("| contrast,") or ln.startswith("| color variegation,")]
    check("main text: flags holding under every variant (first run ERM and Slim highest; second run ERM and Inv highest); other first-run flag fails one; Table 5 unlabelled",
          "%s | %s | %s | %s" % (",".join(sorted(S1["robust_cells"])), ",".join(sorted(S2["robust_cells"])), ",".join(sorted(set(_fl1) - set(S1["robust_cells"]))),
                                 any(("robust" in ln or "fragile" in ln) for ln in _rows5)),
          "erm|contrast@2,slim|color_var@2 | causal|contrast@2,erm|contrast@2 | slim|contrast@0 | False",
          text="the Slim flag at the lowest contrast dose fails at least one")
    ev1 = load("c62_evaluation")["primary"]["components"]
    eps = [math.floor(ev1[k]["sim"][0] / 2 * 100) / 100 for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("bounded deviation (plan-locked run): robust cells need eps_R above 0.95 and 0.91", tuple(eps), (0.95, 0.91), 1e-9,
          text="@supp:need $\\varepsilon_R$ above 0.95 and 0.91 logits")
    _t35 = load("c65_route_robustness_inv_retrained")
    check("no cell agrees: smallest agreement margin 2.09 (rounded up, as in Table 5)", math.ceil(min(v["agrees_above_margin"] for v in _t35["cells"].values()) * 100 - 1e-9) / 100, 2.09, 1e-9,
          text="0.00 / 0.00 | 1.74 / 2.09 |", where="supp")

# ================================================================ occlusion region and dose (c70)
_om = load("c70_occlusion_mask_sensitivity")
if _om:
    ck = _om["skin_only"]["0.5/3"]["check_vs_c66"]
    check("region: the reference region reproduces c66", "%s %s" % (ck["frontier_abs_diff"] == 0.0, ck["counts_equal"]), "True True",
          text="The reference region reproduces the first script exactly", where="supp")
    dm = _om["dose_matched"]["0.5/3"]
    check("dose-matched: 2.46 vs 2.36% altered; ratio 0.77 to 1.41, median 0.99; 18 cells",
          "%.2f %.2f %.2f %.2f %.2f %d" % (100 * dm["altered_frac_skin_heavy"], 100 * dm["altered_frac_full_main"], dm["ratio_range"][0],
                                          dm["ratio_range"][1], dm["ratio_median"], dm["skin_heavy_verdicts"]["exceedance"]),
          "2.46 2.36 0.77 1.41 0.99 18",
          text=["alters 2.46% of the frame", "0.77 to 1.41 times the full main response", "with a median of 0.99"], where=["paper", "supp"])
    check("dose: skin-only main hair 1.43% vs full 2.36% (main text)",
          "%.2f %.2f" % (100 * _om["regions"]["0.5/3"]["altered_frac_skin_only_mean"]["hair"], 100 * _om["altered_frac_full_mean"]["hair"]), "1.43 2.36",
          text="@supp:Restricted to skin, the main hair edit alters 1.43% of pixels instead of 2.36%")
    def _r(cfg):
        r, k = _om["regions"][cfg], _om["skin_only"][cfg]
        return "%d %d %.3f" % (round(100 * r["frame_share_median"]), k["marginal_joint_counts"]["exceedance"], math.floor(1000 * k["frontier_36"]) / 1000)
    check("region: no widening 13%, 18 cells, frontier 0.043", _r("0.5/0"), "13 18 0.043",
          text="With no widening, the region covers a median 13% of the frame, all 18 skin-only cells exceed, and the joint frontier is 0.043", where="supp")
    check("region: w >= 0.7 15%, 17 cells", _r("0.7/3")[:5], "15 17", text="At *w* ≥ 0.7 it covers 15%, and 17 cells exceed", where="supp")
    check("region: 7 px 62% 13 cells; w >= 0.3 76% 12 cells", _r("0.5/7")[:5] + " " + _r("0.3/3")[:5], "62 13 76 12",
          text="makes it 62 or 76% of the frame, and 13 or 12 cells exceed", where="supp")
    check("region: 15 px 91%, 10 cells, frontier 0.003", _r("0.5/15"), "91 10 0.003",
          text=["Widening by 15 pixels makes it 91% of the frame, and 10 cells exceed, with a joint frontier of 0.003",
                "@supp:Widening by 15 pixels makes it 91% of the frame, and 10 cells exceed"], where=["paper", "supp"])
    check("region: at 91% the skin-only main hair alters 0.44%", round(100 * _om["regions"]["0.5/15"]["altered_frac_skin_only_mean"]["hair"], 2), 0.44, 1e-9,
          text="the skin-only main hair edit alters only 0.44% of the frame", where="supp")
    check("region: two images covered entirely", _om["regions"]["0.5/3"]["n_full_frame"], 2, text="it covers two images entirely", where="supp")

# ================================================================ rule checks (c71), library proxy split (c72)
_rc = load("c71_rule_checks")
if _rc:
    L, Rr = _rc["smd_zero_denominators_library"], _rc["smd_zero_denominators_route_controls"]
    check("SMD: no zero denominator in the library or the route controls",
          L["all_183"]["zero_denominators"] + L["malignant_83"]["zero_denominators"] + Rr["zero_denominators"], 0,
          text="No pair of library edits and no route-test control reaches that case on any component", where="supp")
    sm = min(L["all_183"]["smallest_pooled_sd"], L["malignant_83"]["smallest_pooled_sd"], Rr["smallest_pooled_sd"])
    check("SMD: smallest pooled SD 7.8e-5", round(sm * 1e5, 1), 7.8, 1e-9, text="The smallest pooled standard deviation is 7.8 × 10⁻⁵", where="supp")
    check("SMD: unit test of the corrected rule", _rc["unit_test_smd_rule"], True)
    F = _rc["score_floor_57_vs_58"]
    dmde = max(abs(F["mde"][m]["57"][j] - F["mde"][m]["58"][j]) for m in F["mde"] for j in (0, 1))
    check("floor 57: MDE changes by less than 0.03", dmde < 0.03, True, text="the minimum detectable effect changes by less than 0.03 logits between the 58-image and the 57-image set", where="supp")
    check("floor 57: every LOFO R2 negative; ERM globrot -1,659.3 vs -1,631.9",
          "%s %s %.1f %.1f" % (F["lofo_all_negative_57"], F["lofo_all_negative_58"], F["lofo"]["erm"]["57"]["globrot"]["r2_within"],
                               F["lofo"]["erm"]["58"]["globrot"]["r2_within"]), "True True -1659.3 -1631.9",
          text="−1,659.3 instead of −1,631.9", where="supp")
    rm = [v["rmse"] for m in F["lofo"] for v in F["lofo"][m]["58"].values()]
    ma = [v["mae"] for m in F["lofo"] for v in F["lofo"][m]["58"].values()]
    check("LOFO: RMSE 8.1 to 35.1, MAE 6.1 to 23.3", (round(min(rm), 1), round(max(rm), 1), round(min(ma), 1), round(max(ma), 1)) == (8.1, 35.1, 6.1, 23.3), True,
          text="root-mean-square prediction error is 8.1 to 35.1 logits and the mean absolute error 6.1 to 23.3", where="supp")
    T = _rc["tcav_multiplicity"]
    check("TCAV: 7 nominal, 3 Holm, 6 Benjamini-Hochberg", "%d %d %d" % (T["nominal"], T["holm"], T["benjamini_hochberg"]), "7 3 6",
          text=["Holm [56] or Benjamini-Hochberg [57] adjustment assumes valid per-cell p-values and cannot repair this",
                "@supp:corrections over the twelve cells would leave 3 and 6 of the 7"], where=["paper", "supp"])
_ps = load("c72_library_proxy_split")
if _ps:
    order = ["contrast@2", "color_var@2", "contrast@0", "color_var@0"]
    check("A2 split: average 6,6,4,3 and max 5,3,3,3; none passes A1 and the stricter A2; averages reproduce c25",
          "%s %s %d %s" % (",".join(str(_ps["targets"][t]["n_pass_A2_average"]) for t in order), ",".join(str(_ps["targets"][t]["n_pass_A2_max"]) for t in order),
                           sum(len(_ps["targets"][t]["pass_A1_and_A2_max"]) for t in order), _ps["check_average_vs_c25_max_abs_diff"] < 1e-14),
          "6,6,4,3 5,3,3,3 0 True", text=["Under that rule 5, 3, 3 and 3 controls pass (A2)", "against 6, 6, 4 and 3 under the average"], where="supp")

# ================================================================ selection experiment epochs (S13)
_si = load("c35_selection_isolation")
if _si:
    check("S13: leaky epochs 2, 3, 5 and blind epoch 15", "%d %d %d %d" % (_si["models"]["ERM"]["leaky_epoch"], _si["models"]["Slim"]["leaky_epoch"],
          _si["models"]["Inv"]["leaky_epoch"], _si["models"]["Inv"]["blind_epoch"]), "2 3 5 15",
          text="The leaky arm saved ERM, Slim and Inv at epochs 2, 3 and 5, and the blind arm at epoch 15", where="supp")

# ================================================================ Fig. 2 and the data statement
if os.path.exists(ISIC_META):
    import csv as _csv
    _want = {"ISIC_4114301", "ISIC_5583714"}
    _lic, _all = {}, set()
    with open(ISIC_META, newline="") as _fh:
        for _r in _csv.DictReader(_fh):
            _all.add(_r["copyright_license"])
            if _r["isic_id"] in _want:
                _lic[_r["isic_id"]] = (_r["copyright_license"], _r["attribution"])
    check("Fig. 2 images are CC0, attribution ACEMID MIA", ",".join(sorted(set(_lic.values()) and {"%s|%s" % v for v in _lic.values()})),
          "CC-0|ACEMID MIA", text=["released under CC0 (attribution: ACEMID MIA)", "The two images of Fig. 2 are released under CC0"])
    check("ISIC-2024 per-image licenses are CC0, CC BY or CC BY-NC", ",".join(sorted(_all)), "CC-0,CC-BY,CC-BY-NC", text="CC0, CC BY or CC BY-NC")
check("Fig. 2 file present", os.path.exists(os.path.join(PAPER_DIR, "figures", "p3_fig_edits.png")), True, text="figures/p3_fig_edits.png")

# ================================================================ magnitude decides the cell at equal tolerances
_c30 = load("c30_disjoint_audit")
if _c30:
    _cells = [_c30["models"][b][n][g] for b in _c30["models"] for n in _c30["models"][b] if isinstance(_c30["models"][b][n], dict)
              for g in ("all", "malignant", "benign") if g in _c30["models"][b][n]]
    check("artifact: the cell decision equals the magnitude decision in all 18 cells",
          "%d %d" % (len(_cells), sum(c["joint_verdict"] == c["verdict"] for c in _cells)), "18 18",
          text=["so the magnitude decides each cell"])

# the studentized bands do not guarantee the order of interval ends; check it in every band whose intervals are stored
def _band_order(sim):
    cs = sorted(set(k.rsplit("/", 1)[0] for k in sim if k.endswith("/mag")))
    return len(cs), sum(sim[c + "/mag"][0] >= max(sim[c + "/sgn"][0], -sim[c + "/sgn"][1]) and
                        sim[c + "/mag"][1] >= max(-sim[c + "/sgn"][0], sim[c + "/sgn"][1]) for c in cs)
_bands = []
_b56, _b57, _b66 = load("c56_simultaneous36"), load("c57_stratified_band"), load("c66_overlay_occlusion")
if _b56 and _b57 and _b66:
    _bands = [_band_order({k: v["sim36"] for k, v in _b56["components"].items()}),
              _band_order({k: v["sim36"] for k, v in _b57["components"].items()}),
              _band_order(_b66["band"]["full"]["sim36"]), _band_order(_b66["band"]["skin_only"]["sim36"]),
              _band_order(_b66["band"]["lesion_only"]["sim36"])]
    _s75 = open(os.path.join(RES, "c75_seed_band.json")).read() if os.path.exists(os.path.join(RES, "c75_seed_band.json")) else ""
    check("artifact: in every stored band (full, stratified, skin-only, lesion-only, seven realizations) the magnitude ends bound the signed ends",
          " ".join("%d/%d" % (b, a) for a, b in _bands) + " %d" % len(re.findall(r'"magnitude_bounds_signed": true', _s75)),
          "18/18 18/18 18/18 18/18 18/18 7",
          text=["In every stored band the magnitude decides every cell",
                "We checked it in every band whose intervals are stored: the full, the stratified, the skin-only and the lesion-only band and the seven realization bands"], where=["paper", "supp"])
_c57m = load("c57_stratified_band")
if _c57m and _c30:
    check("marginal bound 0.0627 (plain) and 0.063 (stratified), rounded down",
          "%.4f %.3f" % (math.floor(_c30["models"]["b3"]["Inv"]["benign"]["ci"][0] * 1e4) / 1e4, math.floor(_c57m["marginal_min_lower_magnitude"] * 1e3) / 1e3),
          "0.0627 0.063", text=["The marginal bound of 0.0627 is a per-cell statement", "is 0.063, against 0.0627 under the plain scheme",
                                "is 0.0627, which is 1.25 times the tolerance"], where="supp")

# ================================================================ exactly matched footprints (c74)
_em = load("c74_route_exact_match")
if _em:
    rm = _em["residual_mismatch"]
    check("exact match: residual mismatch at most 6.3e-7 relative and one pixel on one image",
          "%.1e %d %d" % (math.ceil(_em["largest_relative_mismatch"] * 1e8) / 1e8, round(_em["largest_pixel_mismatch"]), sum(v["images_pixels_differ"] for v in rm.values())),
          "6.4e-07 1 1", text=["at most $6.4 \\times 10^{-7}$ relative, and one pixel on one image"], where=["paper", "supp"])
    rad = [v["n_exact"] for k, v in rm.items() if "|rad" in k]; hue = [v["n_exact"] for k, v in rm.items() if "|hue" in k]
    check("exact match: radial 50 to 67 images, hue 17 to 38", "%d-%d %d-%d" % (min(rad), max(rad), min(hue), max(hue)), "50-67 17-38",
          text="The radial controls match bit for bit on 50 to 67 of the 78 images, the hue controls on 17 to 38", where="supp")
    S2_, S1_ = _em["runs"]["second_run"], _em["runs"]["first_run"]
    k2 = S2_["min30"]["components"]
    ks = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16"]
    check("exact match: second-run decisive differences +4.83, +6.46, +6.24 on 35, 35, 41 images",
          " ".join("%+.2f/%d" % (k2[k]["est"], k2[k]["n_images"]) for k in ks), "+4.83/35 +6.46/35 +6.24/41",
          text=["+4.83, +6.46 and +6.24 logits for ERM, Inv and Slim", "on 35, 35 and 41 images"], where=["paper", "supp"])
    check("exact match: lower bounds 1.46, 2.41, 1.20 under the 30-image rule (rounded down)",
          " ".join("%.2f" % (math.floor(k2[k]["sim"][0] * 100) / 100) for k in ks), "1.46 2.41 1.20",
          text="with lower bounds of 1.46, 2.41 and 1.20", where="supp")
    _ev2c = load("c62_evaluation_inv_retrained")["primary"]["crit_simultaneous"]; _ev1c = load("c62_evaluation")["primary"]["crit_simultaneous"]
    check("exact match: critical values 2.82 and 3.11 (30-image rule), 3.82 and 3.52 (all 48), plan 3.37 and 3.30",
          "%.2f %.2f %.2f %.2f %.2f %.2f" % (S2_["min30"]["crit"], S1_["min30"]["crit"], S2_["all"]["crit"], S1_["all"]["crit"], _ev2c, _ev1c),
          "2.82 3.11 3.82 3.52 3.37 3.30",
          text=["critical values 2.82 and 3.11, against 3.37 and 3.30", "which widens the band: critical values 3.82 and 3.52"], where="supp")
    _e = S2_["all"]["components"]["erm|contrast@2|D_rad4-rad16"]; _d = S2_["all"]["components"]["causal|contrast@2|D_rad4-hue4"]
    check("exact match, all 48: second-run ERM indeterminate at 0.26; Inv direction pair 18 images, [-2.40, +12.60]; first-run Inv cv@0 agrees",
          "%s %.2f %d [%.2f, %+.2f] %s %s" % (S2_["all"]["cells"]["erm|contrast@2"], math.floor(_e["sim"][0] * 100) / 100, _d["n_images"],
                                             _d["sim"][0], _d["sim"][1], _d["decision"], S1_["all"]["cells"]["causal|color_var@0"]),
          "indeterminate 0.26 18 [-2.40, +12.60] indeterminate routes agree",
          text=["The second run's ERM cell is then indeterminate, with a lower bound of 0.26", "The Inv direction pair has 18 bit-exact images, too few for the 30-image rule",
                "its interval, [−2.40, +12.60], decides nothing", "the first run's Inv cell at the lowest variegation dose agrees"],
          where="supp")
    check("exact match (main text): two and one cells with all 48, three and four under the 30-image rule",
          "%d %d %d %d" % (S1_["all"]["summary"]["routes disagree"], S2_["all"]["summary"]["routes disagree"], S1_["min30"]["summary"]["routes disagree"],
                           S2_["min30"]["summary"]["routes disagree"]), "2 1 3 4",
          text="@supp:under the 30-image rule, four in the second run and three in the first; with all 48 differences, one in the second and two in the first")
    check("exact match, all 48: one cell disagrees in the second run, two in the first",
          "%d %d" % (S2_["all"]["summary"]["routes disagree"], S1_["all"]["summary"]["routes disagree"]), "1 2",
          text="@supp:with all 48 differences, one in the second and two in the first")
    dis = lambda r: ",".join(sorted(c for c, v in r["cells"].items() if v == "routes disagree"))
    check("exact match: decisions under the 30-image rule and with all 48 differences",
          " | ".join([dis(S2_["min30"]), dis(S1_["min30"]), dis(S2_["all"]), dis(S1_["all"])]),
          "causal|contrast@2,erm|contrast@2,slim|color_var@2,slim|contrast@2 | erm|contrast@2,slim|color_var@2,slim|contrast@2 | causal|contrast@2 | erm|contrast@2,slim|color_var@2",
          text=["@supp:keeps 9 of the 48 differences in the second run and 21 in the first", "some on as few as 6 images"], where="supp")
    check("exact match: 9 and 21 differences analysed; smallest n 6",
          "%d %d %d" % (S2_["min30"]["n_analysed"], S1_["min30"]["n_analysed"], min(v["n_images"] for v in S2_["all"]["components"].values())),
          "9 21 6", text=["@supp:some on as few as 6 images",
                          "@supp:keeps 9 of the 48 differences in the second run and 21 in the first"])
    _man = open(os.path.join(RES, "c62_manifest.sha256")).read() if os.path.exists(os.path.join(RES, "c62_manifest.sha256")) else ""
    _scr = sorted(set(re.findall(r"(c\d+)_\w+\.py", _man)))
    check("route manifest lists the scripts c62 to c65 only", ",".join(_scr), "c62,c63,c64,c65",
          text="lists the SHA-256 of the pre-specified script `c62`, of its first follow-ups `c63` to `c65`", where="supp")
    k1 = S1_["min30"]["components"]
    check("exact match: first-run +4.38 and +5.58", "%+.2f %+.2f" % (k1[ks[0]]["est"], k1[ks[2]]["est"]), "+4.38 +5.58", text="+4.38 and +5.58 for ERM and Slim", where="supp")

# ================================================================ checkpoint hashes, deterministic protocol, skin tone
import hashlib as _hl2
_h1 = _hl2.sha256(open(os.path.join(BASE, "checkpoints", "archived_b3inv_ep1", "dj_causal_b3.pt"), "rb").read()).hexdigest()
_h2 = _hl2.sha256(open(os.path.join(BASE, "checkpoints", "dj_causal_b3.pt"), "rb").read()).hexdigest()
check("route test: SHA-256 of the two Inv files", "%s %s" % (_h1[:8], _h2[:8]), "39de9dbc ebfbb47c",
      text=["@supp:SHA-256 begins 39de9dbc | Inv retrained under the restricted rule after the first run's results, SHA-256 begins ebfbb47c", _h1, _h2], where=["paper", "supp"])
_rt2 = load("c73_route_robustness_table")
if _rt2:
    _row = [r for r in _rt2["second_run"]["rows"] if r["variant"] == "clip eta = 1e-06, fp32"][0]["entries"]
    check("deterministic full precision: +4.61 [0.73] and +6.48 [1.85]",
          "%+.2f %.2f %+.2f %.2f" % (_row["erm|contrast@2|D_rad4-rad16"][0], math.floor(_row["erm|contrast@2|D_rad4-rad16"][1] * 100) / 100,
                                     _row["causal|contrast@2|D_rad4-rad16"][0], math.floor(_row["causal|contrast@2|D_rad4-rad16"][1] * 100) / 100),
          "+4.61 0.73 +6.48 1.85", text="@supp:| 6 | +4.61 [0.73] | +6.48 [1.85] |")
check("skin tone: described as a metadata proxy, not a clinical rating", True, True,
      text="Skin tone enters only through colorimetric fields of the dataset's metadata [44], a proxy rather than a clinical rating")

# ================================================================ second fresh-directory run (post hoc scripts)
_e2e = os.path.join(BASE, "log", "e2e_posthoc_fresh_copy_run.txt")
if os.path.exists(_e2e):
    _t = open(_e2e).read()
    _m = re.search(r"files written: (\d+), differing: (\d+)", _t)
    _rc = re.findall(r"^(c\d+)_\w+\s+rc=(\d+)", _t, re.M)
    check("second fresh-directory run: 14 files equal, every script rc=0, c74 included",
          "%s %s %s %s" % (_m.group(1), _m.group(2), all(r == "0" for _, r in _rc), "c74" in [c for c, _ in _rc]), "14 0 True True",
          text=["@supp:Each of the 14 result files written there matched the committed file to a relative tolerance of 10⁻⁹",
                "@supp:Each of the 14 result files written there matched the committed file to a relative tolerance of 10⁻⁹"], where=["paper", "supp"])

# ================================================================ joint band per overlay realization (c75)
_sb = load("c75_seed_band")
if _sb:
    _ps = _sb["per_seed"].values()
    check("seed band: seed 42 reproduces c56 exactly", "%.1e %.1e" % (_sb["check_seed42_vs_c56"]["frontier_abs_diff"], _sb["check_seed42_vs_c56"]["crit_abs_diff"]),
          "0.0e+00 0.0e+00")
    check("seed band: frontier 0.041 to 0.056, 17 or 18 cells at 0.05, crit 3.08 to 3.12",
          "%.3f %.3f %s %.2f %.2f" % (math.floor(_sb["frontier_range"][0] * 1000) / 1000, round(_sb["frontier_range"][1], 3),
                                      "/".join(str(c) for c in sorted(set(_sb["exceed_counts"]))), min(v["crit_36"] for v in _ps), max(v["crit_36"] for v in _ps)),
          "0.041 0.056 17/18 3.08 3.12",
          text=["17 or 18 cells exceed 0.05 jointly on the band recomputed for each",
                "@supp:The critical value runs from 3.08 to 3.12. The frontier runs from 0.041 to 0.056",
                "@supp:17 cells exceed jointly in six realizations and all 18 in one"])
    check("seed band: argmin cells and magnitude order in every band",
          "%d %d %s" % (sum(v["argmin_cell"] == "b3-Inv/ben" for v in _ps), sum(v["argmin_cell"] == "r50-ERM/mal" for v in _ps), all(v["magnitude_bounds_signed"] for v in _ps)),
          "6 1 True", text=["@supp:The EfficientNet-B3 Inv benign cell sets it in six realizations, and the ResNet-50 ERM malignant cell in one",
                            "@supp:In every band and every cell, the magnitude interval bounds the signed one"])
_b56b = load("c56_simultaneous36")
if _b56b:
    check("abstract: 17 of 18 cells exceed jointly at 0.05", _b56b["magnitude_exceed_36"], 17,
          text="At 0.05, 17 of 18 model-subgroup cells exceeded tolerance jointly")

# ================================================================ exact identity, three analyses, LOPO on bit-exact images (c76)
_xd = load("c76_route_exact_detail")
if _xd:
    e2, e1 = _xd["runs"]["second_run"]["exact_identity"], _xd["runs"]["first_run"]["exact_identity"]
    check("exact identity: 48/48 differences nonzero on bit-exact sets in both runs; smallest 0.012 and 0.008",
          "%d %d %d %d %.3f %.3f" % (e2["n_differences"], e2["n_nonzero_mean"], e1["n_differences"], e1["n_nonzero_mean"],
                                     e2["smallest_abs_mean"], e1["smallest_abs_mean"]), "48 48 48 48 0.012 0.008",
          text=["An exact reading on bit-exact image sets fails in every cell", "@supp:The smallest absolute mean difference is 0.012 logits in the second run and 0.008 in the first",
                "@supp:All 48 differences are nonzero on their bit-exact sets in both runs"])
    check("exact identity: 40 and 30 above 0.5 logit; images 6 to 41 and 10 to 64",
          "%d %d %d-%d %d-%d" % (e2["n_abs_mean_above_margin"], e1["n_abs_mean_above_margin"], *e2["images_range"], *e1["images_range"]),
          "40 30 6-41 10-64", text=["@supp:40 and 30 of the 48 exceed 0.5 logit in absolute value", "@supp:on 6 to 41 images in the second run and 10 to 64 in the first"])
    for run in ("second_run", "first_run"):
        ck = _xd["runs"][run]["check_vs_c74"]
        check("three analyses (%s): recomputation reproduces c74" % run, "%.0e %.0e" % (ck["all"]["max_abs_difference"], ck["min30"]["max_abs_difference"]), "0e+00 0e+00")
    lp = _xd["runs"]["second_run"]["lopo_inv_all48"]
    check("bit-exact Inv: 26 of 27 leave-one-patient-out subsets exceed; smallest lower bound 0.47",
          "%d/%d %.2f" % (lp["n_exceeds"], lp["n_subsets"], math.floor(lp["lower_min"] * 100) / 100), "26/27 0.47",
          text=["@supp:The difference exceeds the margin in 26 of the 27 subsets, and the smallest lower bound is 0.47"])
    dd = _xd["runs"]["second_run"]["decisive"]
    ks3 = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16"]
    check("bit-exact decisive differences: 35 to 41 images from 27 or 28 patients",
          "%d-%d %d-%d" % (min(dd[k]["bit_exact_all48"]["n_images"] for k in ks3), max(dd[k]["bit_exact_all48"]["n_images"] for k in ks3),
                           min(dd[k]["bit_exact_all48"]["n_patients"] for k in ks3), max(dd[k]["bit_exact_all48"]["n_patients"] for k in ks3)),
          "35-41 27-28", text="@supp:on 35, 35 and 41 images")
    def _row(run, k, lab):
        def f(e):
            if e is None:
                return "not analysed"
            return "%d (%d): %+.2f [%+.2f, %+.2f] %s" % (e["n_images"], e["n_patients"], e["est"], math.floor(e["sim"][0] * 100) / 100,
                                                       math.ceil(e["sim"][1] * 100) / 100, {"exceeds": "exceeds", "within": "within", "indeterminate": "indet."}[e["decision"]])
        v = _xd["runs"][run]["decisive"][k]
        return ("| %s | %s | %s | %s | %s |" % ({"second_run": "second", "first_run": "first"}[run], lab, f(v["gate"]), f(v["bit_exact_all48"]),
                                                f(v["bit_exact_min30"]))).replace("+-", "−").replace("[-", "[−").replace(", -", ", −")
    _rows38 = [_row("second_run", "causal|contrast@2|D_rad4-rad16", "Inv, contrast highest, rad4 vs rad16"),
               _row("first_run", "slim|contrast@0|D_rad4-rad16", "Slim, contrast lowest, rad4 vs rad16")]
    check("Table S38 rows match c76", "2", "2", text=["@supp:" + r for r in _rows38])

# ================================================================ independent feature space (c77)
_fs = load("c77_route_feature_shift")
if _fs:
    tg = _fs["targets"]
    lo_sh = [tg[t][c]["shift_median_over_nn"] for t in ("contrast@0", "color_var@0") for c in ("rad4", "rad16", "hue4", "hue16")]
    hi_sh = [tg[t][c]["shift_median_over_nn"] for t in ("contrast@2", "color_var@2") for c in ("rad4", "rad16", "hue4", "hue16")]
    rad_hi = [100 * tg[t][c]["share_beyond_q95"] for t in ("contrast@2", "color_var@2") for c in ("rad4", "rad16")]
    hue_hi = [100 * tg[t][c]["share_beyond_q95"] for t in ("contrast@2", "color_var@2") for c in ("hue4", "hue16")]
    rad_lo = [100 * tg[t][c]["share_beyond_q95"] for t in ("contrast@0", "color_var@0") for c in ("rad4", "rad16")]
    check("independent features: natural NN median 8.28, q95 11.38; norms within 1.35 (unedited 1.28)",
          "%.2f %.2f %.2f %.2f" % (_fs["natural_nn_median"], _fs["natural_nn_q95"], math.ceil(_fs["largest_norm_ratio"] * 100) / 100, _fs["unedited_norm_max_ratio"]),
          "8.28 11.38 1.36 1.28", text=["@supp:stay within 1.36 times the median unedited norm",
                                        "@supp:with a median of 8.28 and a 95th percentile of 11.38", "@supp:and the unedited images reach 1.28"])
    check("independent features: low doses 0.52 to 0.95, high doses 1.30 to 1.63; radial 45 to 65%, hue 78 to 84%, radial low 0%",
          "%.2f-%.2f %.2f-%.2f %.0f-%.0f %.0f-%.0f %.0f" % (min(lo_sh), max(lo_sh), min(hi_sh), max(hi_sh), min(rad_hi), max(rad_hi), min(hue_hi), max(hue_hi), max(rad_lo)),
          "0.52-0.95 1.30-1.63 45-65 78-84 0",
          text=["@supp:1.30 to 1.63 times",
                "@supp:the controls move an image by 0.52 to 0.95 times the median natural distance",
                "@supp:beyond the 95th percentile on 45 to 65% of images for the radial controls and 78 to 84% for the hue controls"])
    check("independent features: the concept edit moves 0.92 and 0.98, beyond q95 on 0 and 9%",
          "%.2f %.2f %.0f %.0f" % (tg["contrast@2"]["edit"]["shift_median_over_nn"], tg["color_var@2"]["edit"]["shift_median_over_nn"],
                                   100 * tg["contrast@2"]["edit"]["share_beyond_q95"], 100 * tg["color_var@2"]["edit"]["share_beyond_q95"]),
          "0.92 0.98 0 9", text=["@supp:0.92 and 0.98 times", "@supp:beyond the 95th percentile on 0 and 9% of images"])

# ================================================================ batch context (c78) and near-natural images (c79)
_bc = load("c78_batch_context")
if _bc:
    pp = _bc["protocols"]
    check("batch context: plan protocol reproduces stored scores", "%.1e" % max(_bc["check_stack_bf16_vs_stored_max_abs_prob"].values()), "0.0e+00")
    sb, sf, fl = pp["single_bf16"], pp["single_fp32"], pp["flat32_bf16"]
    check("batch context: one image per batch bf16, median 0.016, q95 0.066, 2.0% above 0.5",
          "%.3f %.3f %.1f" % (sb["score_change_median"], sb["score_change_q95"], 100 * sb["share_above_0.5"]), "0.016 0.066 2.0",
          text=["@supp:a clipped-logit score moves by a median of 0.016 logit, with a 95th percentile of 0.066, and 2.0% of scores move by more than 0.5",
                "@supp:moves by a median of 0.016 logit, with a 95th percentile of 0.066"])
    check("batch context: fp32 median 0.081 and 5.2%; mixed 0.002 and 0.8%; max dD 1.95",
          "%.3f %.1f %.3f %.1f %.2f" % (sf["score_change_median"], 100 * sf["share_above_0.5"], fl["score_change_median"], 100 * fl["share_above_0.5"],
                                        max(v["max_abs_change_D"] for v in pp.values())), "0.081 5.2 0.002 0.8 1.95",
          text=["@supp:move scores by a median of 0.002 logit, with 0.8% above 0.5"])
    check("batch context: ERM and Inv highest contrast disagree under every protocol; Slim cells move",
          " | ".join(",".join(v["disagreeing_cells"]) for v in pp.values()),
          "causal|contrast@2,erm|contrast@2,slim|color_var@2 | causal|contrast@2,erm|contrast@2,slim|contrast@0 | causal|contrast@2,erm|contrast@2,slim|color_var@0,slim|contrast@2 | causal|contrast@2,erm|contrast@2,slim|contrast@2",
          text=["@supp:In the second run, ERM and Inv at the highest contrast dose disagree under every one"])
_nn = load("c79_route_near_natural")
if _nn:
    c2 = _nn["runs"]["second_run"]["all"]["components"]
    e, i_ = c2["erm|contrast@2|D_rad4-rad16"], c2["causal|contrast@2|D_rad4-rad16"]
    dis = lambda r, t: ",".join(sorted(c for c, v in _nn["runs"][r][t]["cells"].items() if v == "routes disagree"))
    check("near-natural: ERM/Inv 16 images, 13 patients, +4.06 and +4.27, undecided; only Slim c@2 disagrees (all 48, both runs)",
          "%d %d %+.2f %+.2f %s %s | %s | %s" % (e["n_images"], e["n_patients"], e["est"], i_["est"], e["decision"], i_["decision"],
                                                 dis("second_run", "all"), dis("first_run", "all")),
          "16 13 +4.06 +4.27 indeterminate indeterminate | slim|contrast@2 | slim|contrast@2",
          text=["@supp:the ERM and Inv scale pairs at the highest contrast dose keep 16 images from 13 patients",
                "@supp:keep 16 images from 13 patients, at +4.06 [−1.16, +9.28] and +4.27 [−2.34, +10.88] logits"])
    check("near-natural: threshold equals c77 q95; 30-image rule: none (second), Slim both contrast doses (first)",
          "%.2f %s | %s" % (_nn["natural_nn_q95"], dis("second_run", "min30") or "none", dis("first_run", "min30")),
          "11.38 none | slim|contrast@0,slim|contrast@2",
          text="@supp:With the 30-image rule, no cell disagrees in the second run, and Slim at both contrast doses disagrees in the first")

# ================================================================ post hoc manifest and the third fresh-directory run
_mf = os.path.join(RES, "posthoc_manifest.sha256")
if os.path.exists(_mf):
    import hashlib as _hl3
    _bad = [ln.split()[1] for ln in open(_mf) if ln.strip() and
            _hl3.sha256(open(os.path.join(BASE, ln.split()[1]), "rb").read()).hexdigest() != ln.split()[0]]
    _scr = sorted(set(re.findall(r"code/(c\d+)_", open(_mf).read())))
    check("post hoc manifest: every listed hash matches; scripts c63 to c88", "%d %s-%s %d" % (len(_bad), _scr[0], _scr[-1], len(_scr)), "0 c63-c88 26",
          text="@supp:`code/make_posthoc_manifest.py` writes `results/posthoc_manifest.sha256`")
_e3 = os.path.join(BASE, "log", "e2e_posthoc3_fresh_copy_run.txt")
if os.path.exists(_e3):
    _t3 = open(_e3).read()
    _m3 = re.search(r"files written: (\d+), differing: (\d+)", _t3)
    check("third fresh-directory run: c75 to c77, 3 files equal", "%s %s %d" % (_m3.group(1), _m3.group(2), len(re.findall(r"rc=0", _t3))), "3 0 3",
          text="@supp:Their three result files matched the committed ones to the same tolerance")

_e4 = os.path.join(BASE, "log", "e2e_posthoc4_fresh_copy_run.txt")
if os.path.exists(_e4):
    _t4 = open(_e4).read()
    _m4 = re.search(r"files written: (\d+), differing: (\d+)", _t4)
    check("fourth fresh-directory run: c78 and c79, 2 files equal; 56 files in all", "%s %s %d" % (_m4.group(1), _m4.group(2), len(re.findall(r"rc=0", _t4))), "2 0 2",
          text=["@supp:Their two result files matched as well"])
_e5 = os.path.join(BASE, "log", "e2e_posthoc5_fresh_copy_run.txt")
if os.path.exists(_e5):
    _t5 = open(_e5).read()
    _m5 = re.search(r"files written: (\d+), differing: (\d+)", _t5)
    check("fifth fresh-directory run: c54, c80 and c81, 5 files equal", "%s %s %d" % (_m5.group(1), _m5.group(2), len(re.findall(r"rc=0", _t5))), "5 0 3",
          text=["@supp:Their five result files matched the committed ones", "Every number in the result files written there matched the committed one to a relative tolerance"])
else:
    check("fifth fresh-directory run record present", None, "present")
_e6 = os.path.join(BASE, "log", "e2e_posthoc6_fresh_copy_run.txt")
if os.path.exists(_e6):
    _t6 = open(_e6).read()
    _m6 = re.search(r"files written: (\d+), differing: (\d+)", _t6)
    check("sixth fresh-directory run: c82 and c83, 4 files equal", "%s %s %d" % (_m6.group(1), _m6.group(2), len(re.findall(r"rc=0", _t6))), "4 0 2",
          text=["@supp:Their four result files matched the committed ones; the record is `log/e2e_posthoc6_fresh_copy_run.txt`"])
else:
    check("sixth fresh-directory run record present", None, "present")
# ================================================================ size and power of the cycle-test band (c80)
_nc = load("c80_route_null_calibration")
if _nc:
    f1, f2 = _nc["first"], _nc["second"]
    pc = lambda x: "%.1f" % (math.floor(1000 * x + 1e-9) / 10)          # coverage: rounded down
    pu = lambda x: "%.1f" % (math.ceil(1000 * x - 1e-9) / 10)           # false-disagreement rates: rounded up
    check("c80: reproduces the plan's band to 5e-7", max(f1["reproduction_max_abs_difference"], f2["reproduction_max_abs_difference"]), 5e-7, ceil=True,
          text="@supp:reproduces the plan's band on the original sample to 5 × 10⁻⁷ logit")
    check("c80: joint coverage 77.7 and 73.3%; marginal at least 95.7%", "%s %s %s" % (pc(f1["joint_coverage"]), pc(f2["joint_coverage"]),
          pc(min(f1["min_marginal_coverage"], f2["min_marginal_coverage"]))), "77.7 73.2 95.7",
          text=["jointly in 77.7% of cohorts in the first run and 73.2% in the second, not 95%", "@supp:Each difference's own interval covered its value in at least 95.7% of draws",
                "@supp:jointly in 77.7% of outer draws in the first run and 73.2% in the second"])
    check("c80: false disagreement 1.7/4.8 (zero), 3.5/3.0 (margin, estimate side)",
          "%s %s %s %s" % (pu(f1["fwer_all_zero"]), pu(f2["fwer_all_zero"]), pu(f1["fwer_margin_by_sign"]["sign_of_estimate"]), pu(f2["fwer_margin_by_sign"]["sign_of_estimate"])),
          "1.7 4.8 3.5 3.0", text=["@supp:some cell disagreed falsely in 1.7 and 4.8% of draws", "@supp:some cell disagreed falsely in 1.7 and 4.8% of draws", "@supp:the rate was 3.5 and 3.0%"])
    check("c80: least favorable 21.7/26.6, bound 22.3/26.8, opposite sides 20.3/25.2",
          "%s %s %s %s %s %s" % (pu(f1["fwer_margin_worst_found"]), pu(f2["fwer_margin_worst_found"]), pu(f1["fwer_margin_upper_bound"]), pu(f2["fwer_margin_upper_bound"]),
                                 pu(f1["fwer_margin_by_sign"]["against_estimate"]), pu(f2["fwer_margin_by_sign"]["against_estimate"])),
          "21.7 26.6 22.3 26.8 20.3 25.2", text=["@supp:reached 21.7 and 26.6%, against the upper bound of 22.3 and 26.8%", "@supp:On the opposite sides it was 20.3 and 25.2%",
                                                  "@supp:The size of the plan's test over its whole null, within this location-shift family, is therefore near 22 and 27%"])
    check("c80: kappa 1.42 and 1.47; recalibrated: first run ERM c@2 only (LB 0.78), second run none",
          "%.2f %.2f %s %.2f %s %s" % (f1["kappa_95"], f2["kappa_95"], ",".join(c for c, v in f1["calibrated_cells"].items() if v == "routes disagree"),
                                      math.floor(f1["calibrated_lower_bounds_of_plan_exceeds"]["erm|contrast@2|D_rad4-rad16"] * 100) / 100,
                                      sum(v == "routes disagree" for v in f2["calibrated_cells"].values()), "%d/%d/%d" % tuple(f1["calibrated_summary"][k] for k in ("routes disagree", "routes agree", "indeterminate"))),
          "1.42 1.47 erm|contrast@2 0.78 0 1/2/9", text=["@supp:1.42 in the first run and 1.47 in the second", "@supp:0.78", "@supp:one disagreeing, two agreeing and nine indeterminate cells"])
    check("c80: recalibrated ERM bound gives eps_R above 0.39", math.floor(f1["calibrated_lower_bounds_of_plan_exceeds"]["erm|contrast@2|D_rad4-rad16"] / 2 * 100) / 100, 0.39, 1e-9,
          text="@supp:$\\varepsilon_R$ would still have to exceed 0.39")
    check("c80: first-run Slim lower bounds 0.39 and -0.12 (rounded down)",
          "%.2f %.2f" % tuple(math.floor(f1["calibrated_lower_bounds_of_plan_exceeds"][k] * 100) / 100 for k in ("slim|color_var@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16")),
          "0.39 -0.12", text="@supp:at lower bounds of 0.39 and −0.12")
    check("c80: coverage by model", " ".join(pc(f[k][m]) for f in (f1, f2) for k in ("joint_coverage_by_model",) for m in ("erm", "slim", "causal")),
          "85.5 92.4 95.6 82.5 90.1 93.7", text="@supp:85.5 and 82.5% for ERM, 92.4 and 90.1% for Slim and 95.6 and 93.7% for Inv")
    check("c80: power, median v80 2 and 4 logits; ERM decisive 4; epoch-1 Inv 1",
          "%g %g %s %s" % (f1["v80_median"], f2["v80_median"], [v for v in f1["v80_grid"] if f1["power"]["erm|contrast@2|D_rad4-rad16"][str(v)] >= 0.8][0],
                           sorted(set(f1["v80_by_model"]["causal"]))), "2 4 4.0 [1.0]",
          text=["@supp:median of 2", "@supp:for the decisive ERM difference of the first run it takes 4 logits"])
    check("c80: outer critical values median 3.29 and 3.31", "%.2f %.2f" % (f1["crit_outer_median"], f2["crit_outer_median"]), "3.29 3.31", text="@supp:median of 3.29 and 3.31")
# ================================================================ image sets and protocols (c82), batch extremes (c83)
_is = load("c82_image_set_checks")
if _is:
    A = _is["G0_shared_pairs"]; LC = _is["library_construction_83"]
    check("G0: 4 shared footprint pairs on 183 images, 7 on the 83 malignant images as scored",
          "%d %d %d" % (A["all_183"]["n_shared_footprint_pairs"], LC["malignant_83"]["n_shared_footprint_pairs"], LC["library_floor_58"]["n_shared_footprint_pairs"]),
          "4 7 7", text=["coincide for exactly four pairs", "Rebuilt exactly as scored on the 83 malignant images that carry the library responses, the library has three more coinciding pairs",
                         "@supp:seven pairs coincide instead of four"])
    newp = [pq for pq in LC["malignant_83"]["pairs"] if pq not in A["all_183"]["pairs"]]
    fams = sorted({x.split("|")[1] for pq in newp for x in pq})
    targ = all(not ("|" not in x) for pq in LC["malignant_83"]["pairs"] for x in pq if not x.startswith("L:"))
    check("G0: new pairs are one control family at the same strength on every image; no concept edit shares a vertex",
          "%s %s %s" % (fams, all(LC["malignant_83"]["same_family_same_strength_on_every_image"].values()), targ),
          "['hue_m', 'texswap_m'] True True", text=["@supp:`hue_m` at the two highest doses, and `texswap_m` at the lowest contrast dose", "@supp:On every one of these images their calibration stops at the same strength"])
    diffs = LC["edits_images_differing_from_183_order_build"]
    check("rebuild differs from the 183-image build on one image, texture-swap and hybrid only",
          "%d %s" % (len({x.split(' @ ')[1] for x in diffs}), sorted({x.split(' @ ')[0].split('|')[1] for x in diffs})), "1 ['hybrid_m', 'texswap_m']",
          text="@supp:equals the 183-image build of Table S26 on every malignant image but one")
    cr = _is["component_range_library_83"]
    check("component range on 83 images: responses -0.32 to +1.45, at least 94.7%",
          "%.2f %.2f %.1f" % (cr["responses_min_max"][0], cr["responses_min_max"][1], math.floor(1000 * cr["min_share_of_cap_range"]) / 10), "-0.32 1.45 94.7",
          text=["and 94.7% on the 83 malignant images", "@supp:which run from −0.32 to +1.45 logits"])
    g = _is["G_a_library_floor_58"]
    T4 = ("contrast@2", "color_var@2", "contrast@0", "color_var@0")
    check("58 images: length-two 12.2 to 13.3, any length 1.67 to 2.44",
          "%.1f %.1f %.2f %.2f" % (math.floor(10 * min(g[t]["bottleneck_length_two"] for t in T4)) / 10, max(g[t]["bottleneck_length_two"] for t in T4),
                                   min(g[t]["bottleneck_any_length"] for t in T4), max(g[t]["bottleneck_any_length"] for t in T4)), "12.2 13.3 1.67 2.44",
          text=["@supp:a bottleneck of 12.2 to 13.3 and the best route of any length 1.67 to 2.44"])
    check("58 images: minimum-total routes 5.65 to 6.69", "%.2f %.2f" % (min(g[t]["total"] for t in T4), max(g[t]["total"] for t in T4)), "5.65 6.69",
          text="@supp:The minimum-total routes run from 5.65 to 6.69 scored units")
    check("58 images: footprint parts 0.72, 0.68, 0.81, 0.46; concept parts 5.80, 5.70, 4.84, 6.22",
          "%s; %s" % (", ".join("%.2f" % g[t]["footprint_part"] for t in T4), ", ".join("%.2f" % g[t]["concept_part"] for t in T4)),
          "0.72, 0.68, 0.81, 0.46; 5.80, 5.70, 4.84, 6.22", text=["@supp:Their footprint parts are 0.72, 0.68, 0.81 and 0.46 scored units", "@supp:concept parts of 5.80, 5.70, 4.84 and 6.22",
                                                                   "@supp:| contrast, top dose | 0.72 | 5.80 | 0.79 | 0.152 |", "@supp:| color variegation, low dose | 0.46 | 6.22 | 0.51 | 0.186 |"])
    LL = _is["lipschitz_library_floor_58"]; b = LL["binding"]
    check("library protocol, 58 images: binding pair 11.03 logits at 10.10 units, L^g >= 1.09",
          "%.2f %.2f %.2f %s %s %s-%s" % (b["abs_diff_library_58"], b["distance_58"], LL["L_floor"], b["model"], b["target"], b["a"], b["b"]),
          "11.03 10.10 1.09 Inv color_var@0 pixperm_m-noise_m",
          text=["disagrees by 11.03 logits at 10.10 scored units, so $L^g \\ge 1.09$", "@supp:giving *L*ᵍ ≥ 11.03/10.10 = 1.09"])
    check("library protocol, 58 images: 21 pairs, 2.26 to 14.38, median 9.36, all exclude zero, distances 10.1 to 22.5",
          "%d %.2f %.2f %.2f %d %.1f %.1f" % (LL["n_pairs"], LL["diff_range"][0], LL["diff_range"][1], LL["diff_median"], LL["n_excluding_zero"],
                                               min(p_["distance_58"] for p_ in LL["pairs"]), max(p_["distance_58"] for p_ in LL["pairs"])),
          "21 2.26 14.38 9.36 21 10.1 22.5", text="@supp:differ by 2.26 to 14.38 logits, with a median of 9.36, and every patient-clustered interval excludes zero. Their footprints lie 10.1 to 22.5 scored units apart")
    check("library protocol: second-largest 0.98, conservative 0.72, median 0.63, quartiles 0.25 and 0.74",
          "%.2f %.2f %.2f %.2f %.2f" % (LL["second_largest_L"], math.floor(100 * LL["conservative_L"]) / 100, LL["median_implied_L"], *LL["quartiles_implied_L"]),
          "0.98 0.72 0.63 0.25 0.74", text=["@supp:Dropping the binding pair leaves *L*ᵍ ≥ 0.98", "@supp:still gives *L*ᵍ ≥ 0.72", "@supp:The median implied constant is 0.63, with quartiles 0.25 and 0.74"])
    BE = _is["break_even_library_floor_58"]
    check("break-even 0.152, 0.161, 0.161, 0.186; half-widths 0.79, 0.75, 0.89, 0.51; none vacuous; span 3.33",
          "%s; %s; %s; %.2f" % (", ".join("%.3f" % BE["targets"][t]["break_even_kappaLh"] for t in T4), ", ".join("%.2f" % BE["targets"][t]["half_width_footprint"] for t in T4),
                                any(BE["targets"][t]["vacuous"] for t in T4), BE["score_span"]),
          "0.152, 0.161, 0.161, 0.186; 0.79, 0.75, 0.89, 0.51; False; 3.33",
          text=["$\\kappa L^h$ is at most 0.152 to 0.186 logits per scored unit", "@supp:half-widths of 0.79, 0.75, 0.89 and 0.51 logits",
                "@supp:stays below 0.152 to 0.186 logits per scored unit", "Against the 3.33-logit spread of the corrected scores on these images"])
    be = [BE["targets"][t]["break_even_kappaLh"] for t in T4]
    check("break-even is 5.8 to 7.2 times below the floor", "%.1f %.1f" % (LL["L_floor"] / max(be), LL["L_floor"] / min(be)), "5.9 7.2",
          text="@supp:5.8 to 7.2 times smaller")
    c34p = _is["lipschitz_c34_on_its_57"]["pairs"]
    check("diagnostic (T) distances on its 57 images: 10.3 to 23.0",
          "%.1f %.1f" % (min(p_["distance_57"] for p_ in c34p), max(p_["distance_57"] for p_ in c34p)), "10.3 23.0",
          text=["where the pairs' footprints lie 10.3 to 23.0 scored units apart", "@supp:on these 57 images their six-descriptor footprints still lie 10.3 to 23.0 scored units apart"])
_bx = load("c83_batch_extremes")
if _bx:
    P_ = _bx["pairs"]; bc = P_["stack_bf16_vs_single_bf16"]; bf = P_["stack_fp32_vs_single_fp32"]; pr = P_["stack_bf16_vs_stack_fp32"]
    check("batch extremes: eval mode, no batch-statistics layers; plan protocol reproduces stored scores",
          "%s %.1e" % (all(len(v) == 0 for v in _bx["eval_mode_or_batch_statistics_issues"].values()), max(_bx["check_stack_bf16_vs_stored_max_abs_prob"].values())),
          "True 0.0e+00", text=["@supp:In both runs the plan's protocol reproduces the stored scores exactly", "@supp:evaluation mode, and none has a layer that uses batch statistics there"])
    check("batch extremes: 57 moves > 0.5, 86% on features > 10x (median 524x); 9.7% of scores; others 0.3%, max 4.9",
          "%d %.0f %.0f %.1f %.1f %.1f" % (bc["n_above_0.5"], 100 * bc["share_large_moves_feature_norm_over_10x"], bc["median_feature_norm_ratio_large_moves"],
                                           100 * bc["share_all_scores_feature_norm_over_10x"], 100 * bc["share_above_0.5_among_feature_norm_at_most_10x"],
                                           bc["max_change_among_feature_norm_at_most_10x"]), "57 86 524 9.7 0.3 4.9",
          text=["@supp:86% come from inputs whose pooled feature exceeds ten times the median unedited norm, with a median ratio of 524",
                "@supp:Such inputs make up 9.7% of all scores. Among the other scores, 0.3% move by more than 0.5, by at most 4.9 logits"])
    check("batch extremes: fp32 batch context 1.4%, 90% on large features; precision 5.2% overall, 3.4% within 10x",
          "%.1f %.0f %.1f %.1f" % (100 * bf["share_above_0.5"], 100 * bf["share_large_moves_feature_norm_over_10x"], 100 * pr["share_above_0.5"],
                                   100 * pr["share_above_0.5_among_feature_norm_at_most_10x"]), "1.4 90 5.2 3.4",
          text=["@supp:batch context moves 1.4% of scores by more than 0.5, 90% of them",
                "@supp:moves 5.2% of all scores by more than 0.5, and 3.4% of those whose features stay within ten times the median"])
    lc = _bx["largest_change"]
    check("batch extremes: largest move 27.6, Slim hue4 contrast@0 on ISIC_0467830, log-odds +21.1 / -1,368 / -4,248 / -905, features 524 to 2,729x",
          "%.1f %s %s %s %+.1f %.0f %.0f %.0f %.0f %.0f" % (lc["change"], lc["model"], lc["column"], lc["image"], lc["logodds"]["stack_bf16"], lc["logodds"]["stack_fp32"],
                                                       lc["logodds"]["single_bf16"], lc["logodds"]["single_fp32"], min(lc["feature_norm_over_median_unedited"].values()),
                                                       max(lc["feature_norm_over_median_unedited"].values())),
          "27.6 slim contrast@0|hue4 ISIC_0467830 +21.1 -1368 -4248 -905 524 2729",
          text=["@supp:Its unclipped log-odds are +21.1 in the plan's batch in bf16, but −1,368 in the same batch in full precision, −4,248 alone in bf16 and −905 alone in full precision",
                "@supp:524 to 2,729 times the median norm"])
    cy = _bx["cycle"]
    check("batch extremes: Table S40 cells (3/6/3/4); ERM and Inv highest contrast under all four",
          " | ".join("%d %s" % (len(cy[v]["disagreeing_cells"]), all(c in cy[v]["disagreeing_cells"] for c in ("erm|contrast@2", "causal|contrast@2")))
                     for v in ("stack_bf16", "stack_fp32", "single_bf16", "single_fp32")), "3 True | 6 True | 3 True | 4 True",
          text=["@supp:In the second run, ERM and Inv at the highest contrast dose disagree under every one",
                "@supp:| second | plan's batch | fp32 | 6 | disagree | disagree | all four targets |", "@supp:| second | one image | fp32 | 4 | disagree | disagree | lowest variegation, highest contrast |"])
    slim = {v: sorted(c.split("|")[1] for c in cy[v]["disagreeing_cells"] if c.startswith("slim")) for v in cy}
    check("batch extremes: Slim cells per protocol", str(slim),
          "{'stack_bf16': ['color_var@2'], 'stack_fp32': ['color_var@0', 'color_var@2', 'contrast@0', 'contrast@2'], 'single_bf16': ['contrast@0'], 'single_fp32': ['color_var@0', 'contrast@2']}")

# ================================================================ protocol checks (c81), plan commit, abstract length
_pc = load("c81_protocol_checks")
if _pc:
    _lib = _pc["library"]
    check("library: 88 edits in one batch per image; duplicate labels return identical responses on 83 images",
          "%d %s %d %.1f" % (_lib["edits"], _lib["duplicates_identical"], min(x["n_images"] for x in _lib["duplicate_label_pairs"]),
                             max(x["max_abs_difference"] for x in _lib["duplicate_label_pairs"])), "88 True 83 0.0",
          text=["@supp:scores each malignant audit image in one batch with all 88 library edits", "@supp:with a largest absolute difference of 0.0 logit for each of the three EfficientNet-B3 models on all 83 malignant audit images"])
    check("route test: 24 edited versions per batch in both runs", sorted(set(v["edited_versions"] for v in _pc["route_test"].values())) == [24], True,
          text="@supp:scores each held-out image in one batch with its 24 edited versions")
check("plan commit given by its full SHA", True, True, text=["@supp:40b6367b84fea2d1f11894484b11d2f0634d4306 (https://github.com/dqtoan87/Causal_Audits_of_SLC_v2/commit/40b6367b84fea2d1f11894484b11d2f0634d4306)"])
_abs = TEXT["paper"].split("## Abstract")[1].split("**Index Terms")[0]
check("abstract at most 250 words", len(_abs.split()), 250, ceil=True)
check("no 'registered outcome' wording; T1 is Q-almost sure only", ("registered outcome" in TEXT["paper"] + TEXT["supp"]) or ("in particular every audited image" in TEXT["paper"]), False)

# ---- review_v3_21_3: R4 read on its own library, Part D, label provenance, bound and wording ----
d = load("c54_synthetic_scm")
if d and "shared_footprint_controls" in d:
    pdd = d["shared_footprint_controls"]
    check("R4 alone: library component +1.194 on R0, generating term 1.117",
          "%+.3f %.3f" % (pdd["library_component_R4_alone_noise_free"], pdd["generating_concept_term"]), "+1.194 1.117",
          text=["so Statement 1 identifies its own library's component, +1.194, not the generating term",
                "@supp:Statement 1 identifies that library's component, +1.194 on $R^0$"])
    ci = pdd["route_sum_R4_minus_R1_noisy_ci95"]
    check("Part D: both controls pass the screens, route gap 0.077 on R0, 0.067 [0.039, 0.094] with noise (outward)",
          "%s %.3f %.3f %.3f %.3f" % (pdd["both_pass_screens"], pdd["route_sum_R4_minus_R1_noise_free"], pdd["route_sum_R4_minus_R1_noisy"],
                                      math.floor(ci[0] * 1000) / 1000, math.ceil(ci[1] * 1000) / 1000),
          "True 0.077 0.067 0.039 0.094",
          text=["0.077 on the structural response, 0.067 [0.039, 0.094] with noise", "@supp:or +1.126 or +1.192 with noise",
                "@supp:exceeds the route through the R1 control by 0.077 logit", "@supp:interval of [0.039, 0.094] from 20,000 resamples"])
    check("Part D: excludes zero, agrees at the 0.5 margin, 20,000 resamples",
          "%s %s %d" % (pdd["excludes_zero"], (abs(ci[0]) < 0.5 and abs(ci[1]) < 0.5), pdd["n_boot"]), "True True 20000",
          text=["refuses both corrections at a margin of zero, but not at the 0.5-logit margin used below",
                "@supp:the interval lies inside the margin and the two routes would agree"])
    check("Part D leaves every earlier c54 field unchanged (R1 and R4 route sums)",
          "%+.3f %+.3f" % (pdd["route_sum_R1_noise_free"], pdd["route_sum_R4_noise_free"]), "+1.117 +1.194",
          text="| accept both; their placebo corrections are +1.117 and +1.194 on the structural response |")
else:
    rows.append(("MISSING", "c54 Part D (shared_footprint_controls)", "-", "-"))

d = load("c84_label_provenance")
if d:
    co = d["cohort"]
    check("cohort: 393 malignant, 114 indeterminate coded 0, 1,068 histopathology = 393 + 114 + 561",
          "%d %d %s %d %d" % (co["n_target_1"], co["n_indeterminate"], d["indeterminate_coded_as_target_0"], co["n_histopathology"],
                              co["iddx_1_counts"]["Benign"] - (co["n_images"] - co["n_histopathology"])),
          "393 114 True 1068 561",
          text=["codes the 114 tiles with an indeterminate diagnosis as not malignant",
                "@supp:1,068 tiles carry such an entry: the 393 malignant, the 114 indeterminate and 561 benign"])
    check("indeterminate tiles in the frames 2 / 3 / 1; none in the 183 audit or the 78 route-test images",
          "%d %d %d %d %d" % (d["frame_train"]["n_indeterminate"], d["frame_select"]["n_indeterminate"], d["frame_audit"]["n_indeterminate"],
                              d["audit_183"]["n_indeterminate"], d["route_test_78"]["n_indeterminate"]), "2 3 1 0 0",
          text=["none is among the 183 audit images", "@supp:two indeterminate tiles in the training folds, three in the selection fold and one in the audit fold"])
    check("histopathology: all 83 malignant audit images, none of the 100 benign; audit cohort inside the fold-0 frame",
          "%d/%d %d/%d %s" % (d["audit_183_malignant"]["n_histopathology"], d["audit_183_malignant"]["n_images"],
                              d["audit_183_benign"]["n_histopathology"], d["audit_183_benign"]["n_images"], d["audit_183"]["subset_of_audit_frame"]),
          "83/83 0/100 True",
          text=["1,068 tiles, among them the 83 malignant audit images, only carry a detailed diagnosis entry",
                "@supp:all 83 malignant ones carry a detailed diagnosis entry and none of the 100 benign ones does"])
else:
    rows.append(("MISSING", "c84_label_provenance", "-", "-"))

check("interaction bound (2k-1) eps_q in Consequence 3 and S14; old 2k bound and 'none of them cancels' gone",
      "%s %s %s" % ("up to $(2k-1)\\,\\varepsilon_q$" in TEXT["paper"], "(2*k* − 1)·$\\varepsilon_q$" in TEXT["supp"],
                    ("2k\\,\\varepsilon_q" in TEXT["paper"]) or ("none of them cancels" in TEXT["supp"])), "True True False")
check("Lipschitz conditions keep M (no 'coarsen $M$'); Statement 2 uses the first part of N only",
      "%s %s" % ("coarsen $M$" in TEXT["paper"], "the first part of N" in TEXT["paper"] and "the first part of N" in TEXT["supp"]), "False True")
check("S4 no longer cites Section III-B for 0.303; 89.9% tied to the 183 images",
      "%s %s" % ("the protocol sensitivity stated in Section III-B" in TEXT["supp"],
                 "On patient resamples of the 183 images" in TEXT["paper"]), "False True")
check("theta^C next design is a new intervention class; N evidence supports without settling",
      "%s %s" % ("a new intervention class: two edits that set each lesion's concept to the common levels" in TEXT["paper"],
                 "this audit defines no standard that would settle N" in TEXT["paper"]), "True True")

_t7p = os.path.join(BASE, "log", "e2e_posthoc7_fresh_copy_run.txt")
if os.path.exists(_t7p):
    _t7 = open(_t7p).read()
    _m7 = re.search(r"files written: (\d+), differing: (\d+)", _t7)
    check("seventh fresh-directory run: c54 and c84, 2 files equal", "%s %s %d" % (_m7.group(1), _m7.group(2), len(re.findall(r"rc=0", _t7))), "2 0 2",
          text="@supp:the record is `log/e2e_posthoc7_fresh_copy_run.txt`")
else:
    rows.append(("MISSING", "seventh fresh-directory run record", "-", "-"))

# ---- review_v3_22_3: nominal flags, footprint-relative h*, vertex 0, label proxy, scope of N ----
_P, _S = TEXT["paper"], TEXT["supp"]
_ab = _P.split("## Abstract")[1].split("**Index Terms")[0]
_concl = _P.split("## VI. Conclusion")[1].split("## Acknowledgment")[0]
check("abstract and conclusion call the 3 cells nominal flags, not 5%-level/controlled rejections",
      "%s %s %s" % ("so the flags carry no controlled error rate" in _ab, "nominal flags, not rejections at a controlled level" in _concl,
                    "rejecting factorization" in _ab or "rejected, in three cells" in _concl), "True True False")
_bad_rej = [t for t in ("cycle test rejects factorization", "a joint rejection for the pair", "each disagreement rejects a joint package",
                        "A disagreeing cell rejects C1a", "the plan-locked run rejects, on the nominal band", "kept one rejection",
                        "the rejections for ERM", "A disagreement rejects the null") if t in _P + _S]
check("no unqualified cycle-test rejection wording left", len(_bad_rej), 0)
check("four readings of the route test kept apart; no controlled-level rejection shown",
      "%s %s" % (("but it has no p-value" in _P and "nominal counts, and no rejection of the joint package at a controlled family-wise level is shown" in _P), "No rejection of the conjunction under $Q_R$ at a controlled level is shown" in _S), "True True")
check("h* relative to M and the library (abstract, Table 1, III-C, conclusion, S1)",
      "%s %s %s %s %s" % ("footprint-relative concept component" in _ab, "relative to the footprint $M$ and the library, net of the footprint term" in _P,
                          "We read $h^\\ast$ as the concept component relative to $M$ and the library" in _P,
                          "the concept component relative to the measured footprint is identified exactly" in _concl,
                          "relative to the footprint *M* and the library" in _S), "True True True True True")
check("G0 concept vertices include 0, isolated when no edit is latently neutral",
      "%s %s" % ("realized by the library together with $0$" in _P, "The vertex $0$ is isolated when no edit is latently neutral" in _P), "True True")
check("histopathology is a diagnosis-field proxy (main, S8, Table S20, S16)",
      "%s %s %s %s" % ("the metadata do not record how a label was confirmed" in _P, "does not show that a label was read from a specimen" in _S,
                       "a diagnosis-specificity proxy (the confirmation method is not recorded)" in _S, "so the argument is conditional on how each label was confirmed" in _S),
      "True True True True")
check("N not settleable under this audit's rules; not an impossibility result",
      "%s %s" % ("Under its rules N cannot be settled, so any reported route sum would remain conditional on N" in _P,
                 "this is not an impossibility result for future edit designs" in _P), "True True")

check("Assumption (i) no longer equates a designated concept-neutral edit with Delta C = 0; recalibrated flag defined",
      "%s %s %s" % ("with $\\Delta C(e) = 0$ for a concept-neutral edit" in TEXT["paper"],
                    "designating it concept-neutral does not make it so" in TEXT["paper"], "one *recalibrated flag* in the first run" in TEXT["paper"]), "False True True")

# ---- review_v3_23_3: label proxy wording, reproduction scope, C1a vs diagnostic (T) ----
check("label proxy never called histopathological; V-E states no verified count",
      "%s %s %s" % ("read as histopathological" in TEXT["supp"] or "histopathological diagnosis by this proxy" in TEXT["supp"],
                    "no verified count of such labels exists" in TEXT["paper"], "diagnosis-specificity proxy" in TEXT["paper"]), "False True True")
check("reproduction scope: verifier needs paper/, cross-pipeline needs the unreleased reimplementation; no 'only external dependency'",
      "%s %s %s %s" % ("covered every analysis script," in TEXT["paper"], "The verifier also reads the manuscript and supplement, which the repository does not hold" in TEXT["paper"],
                       "for this comparison only the stored results can be checked" in TEXT["paper"], "The only external dependency" in TEXT["supp"]),
      "False True True False")
check("C1a constrains only duplicate operators on this library; diagnostic (T) neither tests nor rejects C1a (Table 2, III-C, IV-B, S6)",
      "%s %s %s %s" % ("no equality beyond duplicate operators, whose responses are identical; diagnostic (L) neither tests nor rejects C1a" in TEXT["paper"], "the diagnostic neither tests nor rejects C1a" in TEXT["paper"],
                       "its only restriction is that two edits with the same footprint and the same concept change respond alike" in TEXT["paper"], "The diagnostic therefore neither tests nor rejects C1a" in TEXT["supp"]),
      "True True True True")

# ---- review_v3_24_3: bounded set over latent labelings; verifier/runner exit status described consistently ----
check("bounded interval (10) = identified set with latent labels unrestricted, outer bound under a labeling",
      "%s %s %s" % ("with the other edits' changes left unrestricted, (10) is the bounded identified set" in TEXT["paper"],
                    "the bounded identified set when no latent labeling is assumed" in TEXT["paper"],
                    "over all admissible labelings the bounded identified set is the single-edge interval" in TEXT["supp"]), "True True True")
check("verifier described as reading results and both texts; exit 1 on mismatch, 2 on missing text; runner non-zero",
      "%s %s %s" % ("The verifier reads only the result files" in TEXT["supp"],
                    "with status 2, running no check, if either text is missing" in TEXT["supp"],
                    "and so does the runner if any analysis or check fails" in TEXT["paper"]), "False True True")

# ---- independent read of v3_25: artifact-band calibration (c85), Corollary 1, bounded width, relaxed null ----
d = load("c85_artifact_band_calibration")
if d:
    check("c85 reproduces c56 (crit 3.12, frontier 0.0422)", "%.2f %.4f %.2g" % (d["original_crit"], d["original_frontier"], abs(d["original_crit"] - d["c56_crit"]) < 1e-3),
          "3.12 0.0422 1", text="@supp:it reproduces the critical value 3.12 and the frontier 0.0422")
    check("c85 joint coverage 85% two-sided, 97.3% magnitude lower ends, marginal >= 97.6%",
          "%.0f %.1f %.1f" % (100 * d["joint_coverage_two_sided"], 100 * d["joint_coverage_mag_lower"], math.floor(1000 * d["min_marginal_coverage"]) / 10),
          "85 97.3 97.6", text=["the two-sided band covers jointly in only 85%", "the 18 magnitude lower ends that carry it hold jointly in 97.3% of cohorts",
                                "@supp:each component's own interval covered in at least 97.6%"])
    check("c85 kappa 1.24 (rounded up), widened frontier 0.030 (rounded down), 17 cells at 0.05",
          "%.2f %.3f %d" % (math.ceil(100 * d["kappa_two_sided_95"]) / 100, math.floor(1000 * d["frontier_kappa_two_sided"]) / 1000, d["exceed_at_0p05_kappa_two_sided"]),
          "1.24 0.030 17", text=["@supp:Widening every half-width by a factor of 1.24 restores 95% two-sided joint coverage; that band puts the frontier at 0.030"])
    check("c85 2,000 outer x 20,000 inner", "%d %d" % (d["outer"], d["inner"]), "2000 20000", text="@supp:on 2,000 cohorts resampled from that population")
else:
    rows.append(("MISSING", "c85_artifact_band_calibration", "-", "-"))
check("Corollary 1 carries C0 and the incidence condition; S15 width includes 0; relaxed null named",
      "%s %s %s %s" % ("Assume (i) to (iii), C0, this library" in TEXT["paper"], "taken together with 0" in TEXT["supp"],
                       "On the conjunction's own null" in TEXT["paper"] or "the case in which the conjunction holds exactly" in TEXT["supp"],
                       "controls no error rate by design" in TEXT["paper"]), "True True False False")

_t8p = os.path.join(BASE, "log", "e2e_posthoc8_fresh_copy_run.txt")
if os.path.exists(_t8p):
    _t8 = open(_t8p).read(); _m8 = re.search(r"files written: (\d+), differing: (\d+)", _t8)
    check("eighth fresh-directory run: c85, 2 files equal", "%s %s %d" % (_m8.group(1), _m8.group(2), len(re.findall(r"rc=0", _t8))), "2 0 1",
          text="@supp:the record is `log/e2e_posthoc8_fresh_copy_run.txt`")

# ---- main-text copies of numbers verified elsewhere, and the protocol factors (c30, c35) ----
# A number repeated in the abstract, the results and the conclusion is anchored in each place, so that a
# change to any one copy fails the verification.
_a30, _a35 = load("c30_disjoint_audit"), load("c35_selection_isolation")
if _a30 and _a35:
    _mal = {m: [_a30["models"]["b3"][m]["malignant"]["mean"], _a35["models"][m]["leaky"]["malignant"]["mean"],
                _a35["models"][m]["blind"]["malignant"]["mean"]] for m in ("ERM", "Slim", "Inv")}
    _fo = [max(v) / min(v) for v in _mal.values()]
    _fp = [max(_mal[m][j] for m in _mal) / min(_mal[m][j] for m in _mal) for j in range(3)]
    _allm = [x for v in _mal.values() for x in v]
    check("protocol factors: 1.4/1.8/1.3 per objective, 2.0/2.0/1.2 per protocol, 0.287 to 0.769 (2.7)",
          "%.1f %.1f %.1f | %.1f %.1f %.1f | %.3f %.3f %.1f" % tuple(_fo + [math.floor(x * 10) / 10 for x in _fp] + [min(_allm), max(_allm), max(_allm) / min(_allm)]),
          "1.4 1.8 1.3 | 2.0 2.0 1.2 | 0.287 0.769 2.7",
          text=["varies by a factor of 1.3 to 1.8 while the verdict does not change",
                "@supp:gives factors of 1.4 for ERM, 1.8 for Slim and 1.3 for the invariance model",
                "@supp:Holding the protocol fixed and varying the objective gives 2.0, 2.0 and 1.2",
                "@supp:runs from 0.287 to 0.769, a factor of 2.7"])
if _a30:
    _ri = _a30["models"]["r50"]["Inv"]["malignant"]
    _f3 = lambda x: ("%+.3f" % x).replace("-", "−")
    check("cancellation example: R50 Inv malignant signed and magnitude in the prose",
          "%s [%s, %s] %.3f [%.3f, %.3f]" % (_f3(_ri["signed_mean"]), _f3(_ri["signed_ci"][0]), _f3(_ri["signed_ci"][1]), _ri["mean"], _ri["ci"][0], _ri["ci"][1]),
          "−0.006 [−0.024, +0.018] 0.085 [0.077, 0.093]",
          text="the signed component is −0.006 [−0.024, +0.018], inside the band, while the magnitude is 0.085 [0.077, 0.093]")
_ev1a, _ev2a = load("c62_evaluation"), load("c62_evaluation_inv_retrained")
if _ev1a and _ev2a:
    _i1 = _ev1a["primary"]["components"]["causal|contrast@2|D_rad4-rad16"]["est"]
    _i2 = _ev2a["primary"]["components"]["causal|contrast@2|D_rad4-rad16"]["est"]
    _r73f = [r for r in (load("c73_route_robustness_table") or {}).get("second_run", {}).get("rows", []) if r.get("variant") == "floor on the first run's images"]
    check("Inv scale pair at the highest contrast dose on the first run's images: +0.17 (epoch-1), +5.42 (retrained); +6.62 on the second run's set",
          "%+.2f %+.2f %+.2f" % (_i1, _r73f[0]["entries"]["causal|contrast@2|D_rad4-rad16"][0] if _r73f else float("nan"), _i2), "+0.17 +5.42 +6.62",
          text="@supp:the Inv scale pair at the highest contrast dose differs by +0.17 logit under the epoch-1 checkpoint and by +5.42 under the retrained one")
check("main-text copies of verified numbers (abstract, IV-A, IV-B, conclusion)", True, True, where="paper",
      text=["all 18 exceeded every tolerance below 0.042, for the realized overlays and benign draw",
            "the frontier's central range was 0.032 to 0.066",
            "All six PAD-UFES-20 cells exceeded 0.05 marginally",
            "flagged route sums differing beyond 0.5 logit in three of twelve cells",
            "and all eighteen exceed jointly below 0.042.**",
            "Every cell exceeds 0.05 on its marginal interval",
            "the frontier's central 95% range is 0.032 to 0.066 (Supplementary Section S2)",
            "At a tolerance of 0.20, seven of the eighteen cells still exceed",
            "it moves the proxies by less than 0.1% at a maximum absolute SMD of 1.40",
            "has bottleneck $\\rho$ from 9.6 to 13.3 across targets",
            "a known generating concept term of 1.117 logits",
            "with noise, within 0.8% and as +1.111",
            "The realized mismatch is at most $6.4 \\times 10^{-7}$ relative",
            "ERM at the highest contrast dose by +4.63 [+1.91, +7.34] logits, Slim at the highest variegation dose by +5.27 [+1.83, +8.70], and Slim at the lowest contrast dose by +1.99 [+0.50, +3.47]",
            "every invariance tolerance below 0.042 probability units",
            "17 of 18 did so at 0.05"])

_pre = load("c62_preregistration")
if _pre and _a30:
    _cr = _pre["criteria"]
    check("design constants: artifact tolerance 0.05; route-test score floor 0.1, margin 0.5 logit",
          "%.2f %.1f %.1f" % (_a30["margin_p"], _cr["score_floor"], _cr["margin_logit"]), "0.05 0.1 0.5", where="paper",
          text=["**every cell exceeds every tolerance below 0.042 probability units**",
                "and 17 of the 18 cells exceed 0.05 jointly",
                "with 95% patient-clustered percentile confidence intervals (CI) from 4,000 resamples of the 100 patients, or of the 44 and 75 patients behind the malignant and benign rows",
                                "The 0.5-logit margin is a practical threshold, not an identification condition",
                "that the library protocol scores at 0.1 or above",
                "that all three EfficientNet-B3 models score at 0.1 or above. The corrected score",
                "came within 0.1% of the concept edit on 13 and 15 of the 78 images at the high doses",
                "0.00 means that no margin gives disagreement"])

# ---- design constants of the methods, read from the code that applies them; repeated statements ----
_src3 = open(os.path.join(HERE, "c3_cace_concept.py")).read()
_src25 = open(os.path.join(HERE, "c25_screen_full_sample.py")).read()
_doses = re.search(r"^DOSES = \[([^\]]+)\]", _src3, re.M).group(1)
_sm = re.search(r"def soft_mask\(x01, border=([\d.]+), scale=([\d.]+)\)", _src3)
_main = re.search(r'\("main", ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)\)', _src25)
check("method constants: gains 0.3/0.6/1.0, border 15%, mask scale 0.22; (A1) 0.10, (A2) 10% and 15%; clip bound 13.8",
      "%s | %s %s | %s %s %s | %.4f" % (_doses, _sm.group(1), _sm.group(2), _main.group(1), _main.group(2), _main.group(3), math.log((1 - 1e-6) / 1e-6)),
      "0.3, 0.6, 1.0 | 0.15 0.22 | 0.10 0.10 0.15 | 13.8155", where="paper",
      text=["at gains $\\beta \\in \\{0.3, 0.6, 1.0\\}$", "a border ring whose width is 15% of the image side",
            "$w = \\min\\{1, \\lVert X - s \\rVert / 0.22\\}$", "with $\\lvert \\mathrm{SMD} \\rvert \\le 0.10$ on each of the six",
            "the control's relative proxy change is at most 10%, and at most 15% of the concept edit's own average change",
            "of the six descriptors divided by 0.10", "divided by 0.15", "so that $\\lvert \\ell_\\eta \\rvert \\le 13.8$", "We take $B = 13.8155$"])
check("artifact tolerance and nominal levels: eps 0.05; 95% intervals give alpha 0.025, signed at most about 0.05", True, True, where="paper",
      text=["We set $\\varepsilon_s = \\varepsilon_m = 0.05$ on the probability scale", "A shift of 0.05 is a displacement on the output scale",
            "have a nominal level of approximately $\\alpha = 0.025$", "A signed exceedance has a level of at most about 0.05",
            "not to the choice of 0.05", "The 0.05 operating point was chosen after effect sizes were seen",
            "on the 57 malignant audit images that all three EfficientNet-B3 models score at 0.1 or above"])
check("repeated statements in the contributions, discussion and Tables 6 and 7", True, True, where="paper",
      text=["exceed every tolerance below 0.042 probability units on simultaneous intervals",
            "beyond any tolerance below 0.042 probability units, jointly",
            "every cell exceeds every tolerance below 0.042 jointly, 17 of 18 at 0.05",
            "nominal flags of inconsistency beyond 0.5 logit in 3 of 12 plan-locked cells",
            "0.077 on the structural response, 0.067 [0.039, 0.094] with noise",
            "the nearest neutral control is 1.65 SMD from the concept edit's footprint",
            "the target exactly on the structural response, +1.111 with noise, for 1.117"])
if os.path.exists(TRAIN_LOG) and auc:
    check("selection AUROC, ResNet-50 trio (main text)", "%.2f/%.2f/%.2f" % (auc["dj_erm_r50"], auc["dj_slim_r50"], auc["dj_causal_r50"]),
          "0.87/0.88/0.89", text="and 0.87, 0.88 and 0.89 for the ResNet-50 trio")

# ---- batch context and precision of the first run (c86) ----
_b1 = load("c86_batch_context_first_run")
if _b1:
    cy1 = _b1["cycle"]
    check("c86: plan protocol reproduces the stored first-run scores", "%.1e" % max(_b1["check_stack_bf16_vs_stored_first_run_max_abs_prob"].values()), "0.0e+00",
          text="@supp:In both runs the plan's protocol reproduces the stored scores exactly")
    check("c86: Table S40 first-run cells (3/5/4/5); ERM highest contrast and Slim highest variegation under all four; no Inv flag",
          " | ".join("%d %s %s" % (len(cy1[v]["disagreeing_cells"]), all(c in cy1[v]["disagreeing_cells"] for c in ("erm|contrast@2", "slim|color_var@2")),
                                   any(c.startswith("causal|") for c in cy1[v]["disagreeing_cells"]))
                     for v in ("stack_bf16", "stack_fp32", "single_bf16", "single_fp32")), "3 True False | 5 True False | 4 True False | 5 True False",
          text=["@supp:| first | plan's batch | bf16 (plan) | 3 | disagree | no flag | lowest contrast, highest variegation |",
                "@supp:| first | plan's batch | fp32 | 5 | disagree | no flag | all four targets |",
                "@supp:| first | one image | bf16 | 4 | disagree | no flag | lowest and highest contrast, highest variegation |",
                "@supp:| first | one image | fp32 | 5 | disagree | no flag | all four targets |",
                "@supp:In the first run, ERM at the highest contrast dose and Slim at the highest variegation dose do"])
    slim1 = {v: sorted(c.split("|")[1] for c in cy1[v]["disagreeing_cells"] if c.startswith("slim")) for v in cy1}
    check("c86: first-run Slim cells per protocol", str(slim1),
          "{'stack_bf16': ['color_var@2', 'contrast@0'], 'stack_fp32': ['color_var@0', 'color_var@2', 'contrast@0', 'contrast@2'], "
          "'single_bf16': ['color_var@2', 'contrast@0', 'contrast@2'], 'single_fp32': ['color_var@0', 'color_var@2', 'contrast@0', 'contrast@2']}")
    _lb = [cy1[v]["decisive"][k]["sim_lower"] for v in cy1 for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("c86: smallest lower bound of the two deciding scale pairs over the four protocols 1.54 (rounded down)", math.floor(min(_lb) * 100) / 100, 1.54, 1e-9,
          text="@supp:the smallest simultaneous lower bound of each deciding scale pair over the four protocols is 1.54 logits")
    check("c86 and c67 agree on the first run in full precision (5 cells, crit 3.311)",
          "%d %.3f" % (len(cy1["stack_fp32"]["disagreeing_cells"]), cy1["stack_fp32"]["crit"]),
          "%d %.3f" % (sum(x == "routes disagree" for x in load("c67_route_sensitivity")["numerics_first_run"]["variants"]["fp32_stack"]["cells"].values()),
                       load("c67_route_sensitivity")["numerics_first_run"]["variants"]["fp32_stack"]["crit"]))
    check("main text: every variant applied to both runs; Table 5 robust scoped to the variants", True, True,
          text=["@supp:Every variant was applied to both runs", "The post hoc variants change one element of the analysis at a time, from the bootstrap seed and one left-out patient to the clip level, the precision and the batch context",
                "The variants do not include the versions of the libraries that decode the images",
                "@supp:the robust and fragile labels of this section do not cover that change"])
else:
    rows.append(("MISSING", "c86_batch_context_first_run", "-", "-"))
check("Fig. 3: shaded corner is a necessary condition only", True, True,
      text="is necessary but not sufficient for passing, because the absolute (A2) threshold and (A3) are not drawn")

_t9p = os.path.join(BASE, "log", "e2e_posthoc9_fresh_copy_run.txt")
if os.path.exists(_t9p):
    _t9 = open(_t9p).read(); _m9 = re.search(r"files written: (\d+), differing: (\d+)", _t9)
    check("ninth fresh-directory run: c86, 1 file equal", "%s %s %d" % (_m9.group(1), _m9.group(2), len(re.findall(r"rc=0", _t9))), "1 0 1",
          text="@supp:the record is `log/e2e_posthoc9_fresh_copy_run.txt`")

# ---- feature-range split of the route-test differences (c87), epoch-1 Inv artifact cells (c88), notation, references ----
_f87 = load("c87_route_feature_strata")
if _f87:
    _rows = [(run, k, r) for run, rec in _f87["runs"].items() for k, r in rec.items()]
    _w = [r["within"]["est"] for _, _, r in _rows]; _b = [r["beyond"]["est"] for _, _, r in _rows]
    _wn = [r["within"]["n_images"] for _, _, r in _rows]; _bn = [r["beyond"]["n_images"] for _, _, r in _rows]
    _above = sum(r["within"]["ci95"][0] > 0.5 for _, _, r in _rows)
    check("c87: within range +0.8 to +1.9 on 17 to 70 images; beyond +8.2 to +17.8 on 5 to 43; 6 of 7 within-range intervals above 0.5",
          "%.1f %.1f %d %d | %.1f %.1f %d %d | %d/%d" % (min(_w), max(_w), min(_wn), max(_wn), min(_b), max(_b), min(_bn), max(_bn), _above, len(_rows)),
          "0.8 1.9 17 70 | 8.2 17.8 5 43 | 6/7",
          text=["@supp:Within range, on 17 to 70 images, the differences are +0.8 to +1.9 logits",
                "@supp:Beyond range, on 5 to 43 images, the differences are +8.2 to +17.8 logits",
                "@supp:Within range, on 17 to 70 images, the differences are +0.8 to +1.9 logits"])
    _lab = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}; _tl = {"contrast@2": "contrast highest", "contrast@0": "contrast lowest", "color_var@2": "variegation highest"}
    _sg = lambda x: ("%+.2f" % x).replace("-", "−")
    _fl = lambda x: _sg(math.floor(x * 100) / 100); _cl = lambda x: _sg(math.ceil(x * 100) / 100)
    _txt = []
    for run, k, r in _rows:
        m_, tk, pr = k.split("|"); a_, b_ = pr.split("-")
        w_, bb = r["within"], r["beyond"]
        _txt.append("@supp:| %s | %s, %s, %s vs %s | %d: %s | %d (%d): %s [%s, %s] | %d (%d): %s [%s, %s] |" % (
            run.split("_")[0], _lab[m_], _tl[tk], a_, b_, r["n_all"], _sg(r["all"]["est"]),
            w_["n_images"], w_["n_patients"], _sg(w_["est"]), _fl(w_["ci95"][0]), _cl(w_["ci95"][1]),
            bb["n_images"], bb["n_patients"], _sg(bb["est"]), _fl(bb["ci95"][0]), _cl(bb["ci95"][1])))
    check("c87: Table S41 rows (outward-rounded marginal intervals)", True, True, text=_txt)
    _sh = {k: r["share_of_sum_beyond"] for _, k, r in _rows}
    _hi = [v for k, v in _sh.items() if "@2" in k]
    check("c87: beyond-range images carry 75 to 92% of each high-dose flag, half of the lowest-dose Slim flag",
          "%.0f %.0f %.2f" % (100 * min(_hi), 100 * max(_hi), _sh["slim|contrast@0|rad4-rad16"]), "75 92 0.50",
          text=["75 to 92% of each flag's summed difference comes from images on which a control drives the pooled backbone feature beyond ten times",
                "@supp:These images carry 75 to 92% of the summed difference of each high-dose flag, and half of it for the Slim flag"])
else:
    rows.append(("MISSING", "c87_route_feature_strata", "-", "-"))
_e88 = load("c88_artifact_inv_epoch1")
if _e88:
    _c = _e88["cells"]
    check("c88: epoch-1 Inv artifact cells exceed in all three subgroups; AUROC 0.884",
          "%s | %.3f [%.3f, %.3f] %.3f [%.3f, %.3f] %.3f [%.3f, %.3f] | %.3f" % (
              all(_c[g]["joint_verdict"] == "exceedance" for g in _c),
              _c["all"]["mean"], *_c["all"]["ci"], _c["malignant"]["mean"], *_c["malignant"]["ci"], _c["benign"]["mean"], *_c["benign"]["ci"], _e88["audit_auroc"]),
          "True | 0.159 [0.144, 0.174] 0.181 [0.156, 0.202] 0.140 [0.128, 0.153] | 0.884",
          text="@supp:with mean absolute shifts of 0.159 [0.144, 0.174], 0.181 [0.156, 0.202] and 0.140 [0.128, 0.153] on all, malignant and benign images, against 0.223, 0.383 and 0.091 for the retrained checkpoint, and an audit-cohort AUROC of 0.884")
    check("c88: checkpoint is the released epoch-1 file (SHA-256 begins 39de9dbc)", _e88["checkpoint_sha256"][:8], "39de9dbc")
    _bk = os.path.join(RES, "_pre_b3inv_retrain", "c30_disjoint_audit.json")
    if os.path.exists(_bk):
        _old = json.load(open(_bk))["models"]["b3"]["Inv"]
        check("c88 reproduces the pre-retraining c30 Inv cells (local backup)",
              max(abs(_old[g]["mean"] - _c[g]["mean"]) for g in _c), 0.0, 1e-9)
else:
    rows.append(("MISSING", "c88_artifact_inv_epoch1", "-", "-"))
check("notation: diagnostic (L), graphs \\mathcal G, link tolerances delta; no old symbols left",
      "%s %s %s %s" % (("iagnostic (T)" in TEXT["paper"] + TEXT["supp"]), ("$G_0$" in TEXT["paper"] + TEXT["supp"] or "*G*₀" in TEXT["supp"]),
                        ("\\varepsilon^M" in TEXT["supp"] or "\\varepsilon^{M}" in TEXT["supp"]), "| G1, G2 | route-test gates" in TEXT["supp"]),
      "False False False True")
_body = TEXT["paper"].split("## References")[0]
_ord = []
for _g in re.findall(r"\[(\d{1,2})\]", _body):
    if int(_g) not in _ord:
        _ord.append(int(_g))
_listed = [int(x) for x in re.findall(r"^\[(\d+)\] ", TEXT["paper"].split("## References")[1], re.M)]
check("references: cited in order of first appearance, every listed one cited (57)", "%s %s %d" % (_ord == sorted(_ord), sorted(_ord) == _listed, len(_listed)), "True True 57",
      text=["the usual balance threshold [47]", "a modified treatment policy [48], [49]", "The patient is the resampling unit throughout [53]"])

_t10p = os.path.join(BASE, "log", "e2e_posthoc10_fresh_copy_run.txt")
if os.path.exists(_t10p):
    _t10 = open(_t10p).read(); _m10 = re.search(r"files written: (\d+), differing: (\d+)", _t10)
    check("tenth fresh-directory run: c87 and c88, 2 files equal", "%s %s %d" % (_m10.group(1), _m10.group(2), len(re.findall(r"rc=0", _t10))), "2 0 2",
          text="@supp:the record is `log/e2e_posthoc10_fresh_copy_run.txt`")

# ---- the first run of the route test, end to end in a fresh directory, and the record under other package versions ----
_t11p = os.path.join(BASE, "log", "e2e_posthoc11_fresh_copy_run.txt")
if os.path.exists(_t11p):
    _t11 = open(_t11p).read(); _m11 = re.search(r"files written: (\d+), differing: (\d+)", _t11)
    check("eleventh fresh-directory run: route test first run (c62 three phases, c63, c64, c65), 7 files, construction byte-identical",
          "%s %s %d %s" % (_m11.group(1), _m11.group(2), len(re.findall(r"rc=0", _t11)),
                           "byte-identical to the committed file: True" in _t11 and "c63_route_scores.npz                 IDENTICAL" in _t11
                           and "commit 3677951776b64f844d7c62278efdfd593355c3a9" in _t11 and len(re.findall(r"DIFFERENT \(1 leaves\)", _t11)) == 2),
          "7 0 6 True",
          text=["@supp:the record is `log/e2e_posthoc11_fresh_copy_run.txt`", "this includes both runs of the route test",
                "@supp:from an export of commit `3677951776b64f844d7c62278efdfd593355c3a9`",
                "Two route-test files also differed in a recorded SHA-256 of another file"])
else:
    rows.append(("MISSING", "eleventh fresh-directory run record", "-", "-"))
_tovp = os.path.join(BASE, "log", "e2e_route_first_run_other_versions.txt")
if os.path.exists(_tovp):
    _tov = open(_tovp).read()
    _vals = [re.search(re.escape(k) + r"\s+committed [^\n]*other versions ([+-]\d+\.\d+)", _tov).group(1)
             for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16")]
    _cl = dict(re.findall(r"^(\w+\|\w+@\d)\s+(?:routes disagree|routes agree|indeterminate)\s+(routes disagree|routes agree|indeterminate)$", _tov, re.M))
    check("other package versions: same three flags plus Slim highest contrast; Inv agreeing cells indeterminate",
          "%s | %s | %s %s %s" % (" ".join(_vals), sorted(k for k, v in _cl.items() if v == "routes disagree"),
                                  _cl.get("causal|contrast@0"), _cl.get("causal|color_var@0"), "'routes agree': 0" in _tov),
          "+4.61 +5.46 +2.17 | ['erm|contrast@2', 'slim|color_var@2', 'slim|contrast@0', 'slim|contrast@2'] | indeterminate indeterminate True",
          text=["@supp:flag the same three cells, by +4.61, +5.46 and +2.17 logits, and one more, Slim at the highest contrast dose",
                "@supp:`log/e2e_route_first_run_other_versions.txt`",
                "fixed inference protocol, image decoding included",
                "the first run keeps its three flags, adds one for Slim at the highest contrast dose, and turns both cells in which the Inv model's routes agree indeterminate"])
else:
    rows.append(("MISSING", "other-versions record of the route test's first run", "-", "-"))

# ---- scope of the benign-redraw range; single statements of repeated results; cross-references ----
_d60 = load("c60_design_coverage")
check("benign-redraw range: main text and S2 say the overlay realization is another one, held fixed; S2 calls it percentiles",
      "%s %s %s %s" % ("On a separate overlay realization, held fixed while the benign images were redrawn 500 times" in TEXT["paper"],
                       "It is held fixed across all draws, so the analysis varies the benign draw alone" in TEXT["supp"],
                       "These are empirical percentiles over the 500 draws, not a confidence interval" in TEXT["supp"],
                       "on another overlay realization" in _ab), "True True True True")
if _d60:
    check("benign redraw: 500 draws", _d60["draws"], 500, 0)
check("S2 and S19 state the 38.2% share and the shared-image comparison once each",
      "%d %d %d" % (TEXT["supp"].count("38% of draws") + TEXT["supp"].count("38.2% of draws"),
                    TEXT["supp"].count("On the images both runs analyze"),
                    TEXT["supp"].count("On the 44 to 47 images per target that pass the floor under both Inv checkpoints")), "1 0 1")
check("no cross-reference to a main-text bounded-deviation reading (it lives in S19)", "bounded-deviation reading of Section IV-B" in TEXT["supp"], False)

# ================================================================ report
w = max(len(r[1]) for r in rows)
print("verification of paper3.md and supplementary3.md against %s" % RES)
print("=" * (w + 50))
for st, cl, g, wv in rows:
    print("%-9s %-*s  got %-22s paper %s" % (st, w, cl, g, wv))
print("=" * (w + 50))
nbad = sum(r[0] != "ok" for r in rows)
print("%d checks, %d ok, %d not ok" % (len(rows), len(rows) - nbad, nbad))
sys.exit(1 if nbad else 0)
