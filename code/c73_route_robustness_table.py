#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C73: one table of the route test's decisive differences across every post hoc variant  (reads results only)

It collects, for the second run, the four differences that decide a cell under the plan, and the count of
disagreeing cells, under each variant computed by c65 (seeds, leave one patient out), c67 (score floor,
numerics) and c69 (clip level, floor read in full precision). For the first run it does the same for the
three differences that decided its cells. Every entry is the estimate and the simultaneous lower bound.
No new number is computed except minima over variants.

Out -> results/c73_route_robustness_table.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c73_route_robustness_table.json")
KEYS2 = ["erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4",
         "slim|color_var@2|D_rad4-rad16"]
KEYS1 = ["erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16", "slim|contrast@0|D_rad4-rad16"]


def load(n):
    return json.load(open(os.path.join(RES, n)))


def is_dis(v):
    if isinstance(v, dict):                   # c62 stores {"class": ..., "pairs": ...}
        v = v["class"]
    return v in ("routes disagree", "disagree")


def n_dis(cells):
    return sum(is_dis(v) for v in cells.values())


def row(name, comps, cells, keys):
    return {"variant": name, "disagree": n_dis(cells),
            "disagreeing_cells": sorted(c for c, v in cells.items() if is_dis(v)),
            "entries": {k: [comps[k]["est"], comps[k]["sim"][0]] for k in keys}}


def table(run):
    keys = KEYS2 if run == "second_run" else KEYS1
    suf = "_inv_retrained" if run == "second_run" else ""
    ev = load("c62_evaluation%s.json" % suf)["primary"]
    c65 = load("c65_route_robustness%s.json" % suf)
    c67 = load("c67_route_sensitivity.json")
    c69 = load("c69_route_clip_sensitivity.json")["runs"][run]
    rows = []
    comps = {k: {"est": v["est"], "sim": v["sim"]} for k, v in ev["components"].items() if "|D_" in k}
    rows.append(row("pre-specified", comps, ev["cells"], keys))
    seeds = {k: [min(s["lower"] for s in c65["decisive"][k]["seeds"]),
                 sum(s["decision"] == "exceeds" for s in c65["decisive"][k]["seeds"]), len(c65["decisive"][k]["seeds"])]
             for k in keys if k in c65["decisive"]}
    lopo = {k: [c65["decisive"][k]["lopo_in_cell_lower_min"], c65["decisive"][k]["lopo_in_cell_exceeds"],
                c65["decisive"][k]["lopo_in_cell_n"]] for k in keys if k in c65["decisive"]}
    if run == "second_run":
        # the second run's floor set lies inside the first run's, so "passed under both" equals the primary set
        for nm, key in (("floor per model", "per_model"), ("floor on the first run's images", "run1_set")):
            r = c67["floor"][key]; rows.append(row(nm, r["components"], r["cells"], keys))
        for nm, key in (("scored in full precision", "fp32_stack"), ("scored in mixed bf16 batches", "bf16_flat32")):
            r = c67["numerics"]["variants"][key]; rows.append(row(nm, r["components"], r["cells"], keys))
    else:
        for nm, key in (("floor per model", "per_model_first_run"),
                        ("floor passed under both Inv checkpoints", "common_set_first_run")):
            r = c67["floor"][key]; rows.append(row(nm, r["components"], r["cells"], keys))
        cells1 = c67["numerics_first_run"]["variants"]
        rows.append({"variant": "scored in full precision", "disagree": n_dis(cells1["fp32_stack"]["cells"]),
                     "disagreeing_cells": sorted(c for c, v in cells1["fp32_stack"]["cells"].items() if is_dis(v)),
                     "entries": None})
        rows.append({"variant": "scored in mixed bf16 batches", "disagree": n_dis(cells1["bf16_flat32"]["cells"]),
                     "disagreeing_cells": sorted(c for c, v in cells1["bf16_flat32"]["cells"].items() if is_dis(v)),
                     "entries": None})
    rf = c69["floor_fp32"]
    rows.append({"variant": "floor read in full precision", "disagree": n_dis(rf["cells"]),
                 "disagreeing_cells": sorted(c for c, v in rf["cells"].items() if is_dis(v)),
                 "entries": {k: [rf["key"][k.rsplit("|", 1)[0]][k.rsplit("|D_", 1)[1]]["est"],
                                 rf["key"][k.rsplit("|", 1)[0]][k.rsplit("|D_", 1)[1]]["sim"][0]] for k in keys}})
    for num in ("bf16", "fp32"):
        for eta, r in c69["eta_" + num].items():
            rows.append(row("clip eta = %s, %s" % (eta, num), r["components"], r["cells"], keys))
    # a disagreeing cell is robust when it disagrees in every variant row and, for at least one of its
    # decisive differences, under all 20 other seeds and in every leave-one-patient-out subset
    def resampling_ok(cell):
        ks = [k for k in keys if k.startswith(cell + "|") and k in seeds]
        return any(seeds[k][1] == seeds[k][2] and lopo[k][1] == lopo[k][2] for k in ks)
    robust = [c for c in rows[0]["disagreeing_cells"] if all(c in r["disagreeing_cells"] for r in rows) and resampling_ok(c)]
    fragile = [c for c in rows[0]["disagreeing_cells"] if c not in robust]
    mins = {}
    for k in keys:
        lows = [r["entries"][k][1] for r in rows if r["entries"]]
        if k in seeds:
            lows += [seeds[k][0], lopo[k][0]]
        mins[k] = min(lows)
    return {"rows": rows, "robust_cells": robust, "fragile_cells": fragile, "seeds": seeds, "leave_one_patient_out": lopo, "smallest_lower_bound": mins,
            "disagree_range": [min(r["disagree"] for r in rows), max(r["disagree"] for r in rows)]}


def main():
    out = {"note": "compiled from c62, c65, c67 and c69; no new computation except minima",
           "second_run": table("second_run"), "first_run": table("first_run")}
    json.dump(out, open(OUT, "w"), indent=1)
    for run in ("second_run", "first_run"):
        t = out[run]
        print("[%s] disagreeing cells across variants: %s; disagree in every variant: %s; smallest lower bounds: %s"
              % (run, t["disagree_range"], t["robust_cells"],
                 {k: round(v, 3) for k, v in t["smallest_lower_bound"].items()}))
        for r in t["rows"]:
            print("   %-40s %d  %s" % (r["variant"], r["disagree"],
                                       {k.split("|")[0] + "|" + k.split("|")[1][:6] + "|" + k.split("D_")[1]:
                                        "%.2f [%.2f]" % tuple(v) for k, v in (r["entries"] or {}).items()}))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
