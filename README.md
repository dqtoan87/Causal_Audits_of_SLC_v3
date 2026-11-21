# Causal_Audits_of_SLC_v3

Analysis code and cached results for a causal audit of skin-lesion classifiers. The audit has two tiers. The artifact tier asks whether a classifier's malignant score moves under performed acquisition edits (hair and ruler overlays), judged against a stated tolerance with a three-way interval rule. The concept tier asks whether a library of performed edits identifies the classifier's response to a clinical concept, decided by the connectivity of an audit graph. A pre-specified route test on held-out images, described at the end of this file, tests the working form behind that decision.


This repository continues [Causal_Audits_of_SLC_v2](https://github.com/dqtoan87/Causal_Audits_of_SLC_v2) with its full history, and adds the manuscript versions in `paper/` (one commit per revision; the commit history gives the sequence).
## Layout

```
code/          analysis scripts, shared modules, the runner and the verifiers
  lib/         vendored training modules
results/       cached JSON and NPZ outputs, one per script, plus the frozen manifests
log/           one log per script; run_retrain_b3inv.log for the last re-run of the analyses;
               e2e_fresh_copy_run.txt and e2e_posthoc*_fresh_copy_run.txt for runs in fresh directories
requirements.txt  exact package versions
checkpoints/   audited checkpoints, downloaded from the release below (not tracked)
paper/         successive versions of the manuscript and its supplement, with their figures
```

Model checkpoints are published as assets of the release [`checkpoints-v1.1`](https://github.com/dqtoan87/Causal_Audits_of_SLC_v3/releases/tag/checkpoints-v1.1), with a `SHA256SUMS` file. Download them into `checkpoints/` at the repository root:

```bash
mkdir -p checkpoints && cd checkpoints
gh release download checkpoints-v1.1 --repo dqtoan87/Causal_Audits_of_SLC_v3
sha256sum -c SHA256SUMS
```

The release holds the six audited classifiers (`dj_{erm,slim,causal}_{b3,r50}.pt`), the two PAD-UFES-20 classifiers (`pad_{erm,slim}.pt`) and the checkpoints of the selection analysis (`sel_{blind,leaky}_{erm,slim,causal}.pt`). Analyses that read only cached predictions run without them (see below).

In `checkpoints-v1.1` the EfficientNet-B3 Inv classifier (`dj_causal_b3.pt`) is selected only among epochs in which both invariance penalties are on (epoch 7). The release [`checkpoints-v1.0`](https://github.com/dqtoan87/Causal_Audits_of_SLC_v3/releases/tag/checkpoints-v1.0) keeps the checkpoint that the original selection rule saved at epoch 1, inside the penalty warm-up. It is needed only for the first run of the route test below and for `c67` and `c69`, which read it from `checkpoints/archived_b3inv_ep1/dj_causal_b3.pt`; its other thirteen files are identical to those of `checkpoints-v1.1`.

```bash
mkdir -p checkpoints/archived_b3inv_ep1 && cd checkpoints/archived_b3inv_ep1
gh release download checkpoints-v1.0 --repo dqtoan87/Causal_Audits_of_SLC_v3 --pattern dj_causal_b3.pt
```

## Setup

Python 3.12 with PyTorch, torchvision, scikit-learn, pandas, numpy, scipy, matplotlib and Pillow. `requirements.txt` pins the exact versions the reported numbers were produced with (PyTorch 2.10.0, CUDA 12.8 build). A CUDA device is needed only by the scripts that run a model over images.

The two datasets are public and are not redistributed here.

| Dataset | Expected at | Set the path in |
| --- | --- | --- |
| ISIC-2024 (SLICE-3D) | `<DATA_ROOT>/data_ISIC2024/train-metadata.csv`, and the images unpacked as `<DATA_ROOT>/data_ISIC2024/train-image/image/<isic_id>.jpg` | `DATA_ROOT` in `code/lib/train_XAI_new_v01.py`; `DATA_CSV` in `code/c1_tier1_ate.py`; `META` in `code/c8_backbone2_train.py`, `code/c29_disjoint_train.py` and `code/c35_selection_isolation.py`; `ISIC_META` in `code/verify_paper3.py` |
| PAD-UFES-20 | `<ROOT>/metadata.csv` and `<ROOT>/imgs_part_1..3/` | `ROOT` in `code/c46_padufes_artifact.py` |

These variables hold the absolute paths of the machine on which the results were produced. Replace them with your own paths before running. Every other path is resolved relative to the repository root, and the route test (`c62` to `c65`) reads the ISIC-2024 images through `DATA_ROOT`.

## Frozen inputs

`results/frozen_split.json`, `results/audit_cohort.json` and `results/padufes_split.json` record the patient-disjoint partition and the audit cohort by image identifier. They are read, never regenerated: fold assignment from a grouped stratified splitter is not stable across library versions, so rebuilding a split without retraining would silently break the patient-disjoint property. `frozen_split.py` and `audit_cohort.py` rebuild them only under an explicit `--rebuild` flag, and `audited_models.py` is the single place that resolves which checkpoint every analysis loads.

## Running

All commands are run from the repository root. Every script resolves its paths relative to its own location and writes into `results/`.

```bash
bash code/run_all.sh                                   # all analyses in dependency order, logs to log/
PY=/path/to/python setsid nohup bash code/run_all.sh > log/nohup.out 2>&1 < /dev/null &   # detached
```

`run_all.sh` audits the released PAD-UFES-20 checkpoints (`c46 --use-saved`), which is how the reported numbers were produced, and runs `c51`, `c58` and `c59` only when `OTHER` points at the reimplementation. It first rebuilds the prediction cache, runs the analysis scripts in dependency order (`c25` before `c45`, `c34` before `c55`), then the post hoc checks (`c66` to `c91`), regenerates the figures, and calls the verifier. A failing script is logged with its traceback and the sequence continues; at the end `run_all.sh` exits with status 1 if any script failed or the verification did not pass, and the last lines of `log/run_all.log` say which. Training is not part of the run: the audited checkpoints are frozen and every reported number is about those weights. To retrain, run `c29_disjoint_train.py` and `c8_backbone2_train.py` first; retrained weights need not reproduce the reported magnitudes. `c29_disjoint_train.py` selects each Inv checkpoint among the epochs with both penalties on, and `C29_ONLY=dj_causal_b3` trains one job only. `c44_cache_preds.py` writes `results/c19_preds.npz`, which `c26`, `c32`, `c38` and the shrinkage figure read, and `results/c19_preds_manifest.json` with the SHA-256 of the checkpoints it read, so a cache built from other weights can be detected. `code/run_retrain_b3inv.sh` records how the EfficientNet-B3 Inv checkpoint was retrained and every analysis that reads it was run again, including the second run of the route test. `log/e2e_fresh_copy_run.txt` records a run of the released code and checkpoints in a fresh directory, compared file by file with `results/`; `log/e2e_posthoc_fresh_copy_run.txt` and `log/e2e_posthoc3_fresh_copy_run.txt` to `log/e2e_posthoc14_fresh_copy_run.txt` record thirteen further fresh-directory runs, each naming the scripts it ran and comparing every result file it wrote with the committed one (relative tolerance 1e-9, timing fields ignored). The eleventh reruns the first run of the route test end to end: the three phases of `c62` with the epoch-1 Inv checkpoint, then `c63`, `c64` and `c65`. Together these runs executed every script that writes a reported result, except the training of the audited checkpoints. The cross-pipeline comparisons `c51`, `c58` and `c59` ran only because the reimplementation was supplied; it is not released, so only their stored results can be checked. Exact agreement needs the package versions in `requirements.txt`: `log/e2e_route_first_run_other_versions.txt` records the same route-test rerun under other Pillow, NumPy and SciPy versions, where the decoded images and hence the result files differ. `make_posthoc_manifest.py` writes `results/posthoc_manifest.sha256`, the SHA-256 of every post hoc script `c63` to `c91` and of its result files.

Scripts that read only cached predictions and need neither a GPU nor raw images include `c26_hair_dose_prob`, `c32_absdp_patient_ci`, `c38_shrinkage_sensitivity`, `c41_within_patient`, `c45_partial_id`, `c47_heterogeneity`, `c54_synthetic_scm`, `c55_identification_graph` and `make_fig_shrinkage`.

## What each script produces

| Script | Output in `results/` | Computes |
| --- | --- | --- |
| `c1_tier1_ate.py` | `tier1_ate.json` | disease side: concept-to-malignancy effects by doubly robust AIPW |
| `c41_within_patient.py` | `c41_within_patient.json` | disease side: conditional logistic regression stratified by patient |
| `c47_heterogeneity.py` | `c47_heterogeneity.json` | common-odds-ratio check for the within-patient model |
| `c29_disjoint_train.py` | `c29_disjoint_split.json` | trains the EfficientNet-B3 trio under the frozen partition; Inv checkpoints are selected among epochs with both penalties on |
| `c8_backbone2_train.py` | checkpoints | trains the ResNet-50 trio |
| `c2_cace_artifact.py` | `tier2_cace_artifact.json` | artifact overlays and edit precomputation |
| `c30_disjoint_audit.py` | `c30_disjoint_audit.json` | artifact criterion on the audit cohort |
| `c40_simultaneous.py` | `c40_simultaneous.json` | max-statistic band over the 18 magnitude statistics (secondary) |
| `c56_simultaneous36.py` | `c56_simultaneous36.json` | max-statistic band over all 36 components, the headline band |
| `c57_stratified_band.py` | `c57_stratified_band.json` | the 36-component band with patients resampled within the design strata |
| `c60_design_coverage.py` | `c60_design_coverage.json` | coverage of the 36-component band over 500 redraws of the 100 benign audit images, with the fold's patients fixed |
| `c32_absdp_patient_ci.py` | `c32_absdp_patient_ci.json` | patient-clustered intervals for the artifact effect |
| `c21_patient_cluster.py` | `c21_patient_cluster.json` | patient-clustered recomputation of the model-side quantities |
| `c26_hair_dose_prob.py` | `c26_hair_dose_prob.json` | dose response on the probability scale |
| `c42_positive_control.py` | `c42_positive_control.json` | dose calibration of the decision rule |
| `c50_overlay_seed.py` | `c50_overlay_seed.json` | stability across independent overlay realizations |
| `c52_overlay_perceptual.py` | `c52_overlay_perceptual.json` | size of each overlay as a change to the image |
| `c19_saturation_mde.py` | `c19_saturation_mde.json` | clipping saturation and minimum detectable effects |
| `c20_numerical_repro.py` | `c20_numerical_repro.json` | fp32 and batch-composition reproducibility |
| `c35_selection_isolation.py` | `c35_selection_isolation.json` | checkpoint-selection effect separated from training-set size |
| `c46_padufes_artifact.py` | `c46_padufes_artifact.json`, `padufes_split.json` | the artifact audit on a second cohort |
| `c3_cace_concept.py` | `tier2_cace_concept.json` | concept edits and the soft lesion mask |
| `c7_extra_controls.py` | `c7_controls.json` | additional placebo controls |
| `c10_multiplacebo.py` | `c10_multiplacebo.json` | placebo families with patient-resampled intervals |
| `c12_matched_control.py` | `c12_matched_control.json`, `c12_preds.npz` | the 19-control library and per-image responses |
| `c13_fullpop.py` | `c13_fullpop.json` | edit responses on the full evaluation cohort |
| `c25_screen_full_sample.py` | `c25_screen_full.json`, `c25_screen_mal.json` | balance, neutrality and overlap screens for every control |
| `c34_ctrl_vs_ctrl.py` | `c34_ctrl_vs_ctrl.json` | control-versus-control diagnostic |
| `c45_partial_id.py` | `c45_bracketing.json` | whether the controls span the concept edit's response |
| `c55_identification_graph.py` | `c55_identification_graph.json` | the audit graph, its connectivity and the approximate-route bounds |
| `c54_synthetic_scm.py` | `c54_synthetic_scm.json` | synthetic response model with a known concept component; Part D (`shared_footprint_controls`) puts two screened controls with one footprint and no concept change in one library, where the additive form fails |
| `c38_shrinkage_sensitivity.py` | `c38_shrinkage_sensitivity.json` | shrinkage slopes with clustered intervals |
| `c44_cache_preds.py` | `c19_preds.npz`, `c19_preds_manifest.json` | cached baseline and edited predictions, with the checksums of the checkpoints they came from |
| `c49_tcav.py` | `c49_tcav.json` | concept activation vectors on the same checkpoints and concepts; accuracy on all activations and on the half not used to fit each direction |
| `c66_overlay_occlusion.py` | `c66_overlay_occlusion.json` | exploratory: the artifact criterion with overlays restricted to skin or to a proxy lesion region |
| `c70_occlusion_mask_sensitivity.py` | `c70_occlusion_mask_sensitivity.json` | exploratory: the skin-only criterion under other lesion regions, and a comparison matched in altered pixels |
| `c68_floor_sets.py` | `c68_floor_sets.json` | the images that pass the 0.1 score floor under each scoring path, by identifier |
| `c71_rule_checks.py` | `c71_rule_checks.json` | zero denominators of the balance SMD, the 57-image floor set, multiplicity of the concept-activation test |
| `c72_library_proxy_split.py` | `c72_library_proxy_split.json` | the proxy-neutrality screen with the two concept proxies kept apart |
| `c75_seed_band.py` | `c75_seed_band.json` | exploratory: the 36-component band and its frontier recomputed for each of seven overlay realizations |
| `c85_artifact_band_calibration.py` | `c85_artifact_band_calibration.json`, `c85_artifact_vectors.npz` | post hoc: size of the 36-component artifact band by a double bootstrap with the 100 audit patients as the population (two-sided joint coverage, joint coverage of the magnitude lower ends, recalibration factor) |
| `c88_artifact_inv_epoch1.py` | `c88_artifact_inv_epoch1.json` | reporting: the artifact criterion of Table 4 for the epoch-1 EfficientNet-B3 Inv checkpoint of `checkpoints-v1.0`, which was retrained after its results had been seen (all three subgroups exceed 0.05) |
| `c89_fullfold_artifact.py` | `c89_fullfold_band.json`, `c89_fullfold_vectors.npz`, `c89_dryrun.json` | post hoc: the 36-component artifact band on the whole audit fold (83 malignant and 4,150 benign images, 199 patients), which removes the second-stage benign draw; overlays as in `c60`; its plan is `c89_plan.json`, fixed before the run |
| `c90_posthoc_rereads.py` | `c90_posthoc_rereads.json` | post hoc rereads of stored results, no model output: which footprint descriptor sets each control's largest imbalance (with CIELAB $\Delta E$ removed from the balance screen as a sensitivity), the Lipschitz floor of the generic term per checkpoint, the size of the cycle test over configurations that route sums can produce, and simulation example B5 on the structural response |
| `c91_feasibility_measured.py` | `c91_feasibility_measured.json` | post hoc, image side only: rebuilds the control library of `c25` (checked against `c25_screen_full.json`) and measures screen (A3) for every control on the final, clipped image, whether its strength comes from construction or calibration; recounts the four threshold schemes |
| `c84_label_provenance.py` | `c84_label_provenance.json` | model-free: how the binary target codes the 114 indeterminate lesions, and how many tiles of each frame carry a detailed diagnosis entry, a diagnosis-specificity proxy (the metadata record no confirmation method) |
| `c51_independent_rebuild.py` | `c51_independent_rebuild.json` | comparison against a separate reimplementation (see the note below) |
| `c58_rebuild_trace.py` | `c58_rebuild_trace.json` | layer-by-layer trace of the one cell on which the reimplementation disagrees (see the note below) |
| `c59_rebuild_generator.py` | `c59_rebuild_generator.json` | reruns the reimplementation's own loader, overlay generator, inference and checkpoints, and compares them with its cached predictions (see the note below) |
| `make_figures_split.py`, `make_fig_edit_examples.py`, `make_fig_shrinkage.py` | `paper/figures/` | figures; `make_fig_edit_examples.py` draws the performed edits on two CC0 images |

The balance SMD of `c12` and `c55` is zero for a zero pooled variance only when the two means agree, and infinite otherwise; `c71` finds no zero denominator in the library or among the route-test controls.

`c15_symmetry.py` and `audit_cohort.py`, `frozen_split.py`, `audited_models.py` are shared modules. `p2_source_guard.py` stops the data-reading figure scripts when the tree lacks its frozen manifest.

`c51_independent_rebuild.py`, `c58_rebuild_trace.py` and `c59_rebuild_generator.py` compare this pipeline with a separate clean-room reimplementation of the same audit. They read that reimplementation's files from a sibling directory, passed with `--other`, which is not part of this repository, so they do not run on their own here. Their cached outputs are included.

`verify_paper5.py` checks the results stated in the main text of the current manuscript, each copy of them, and the supplementary numbers they rest on against the files in `results/`; many descriptive entries of the supplementary tables are not compared. It reads `paper/paper5.md` and `paper/supplementary5.md`, which are part of this repository, or the texts in `--paper-dir DIR` (or `PAPER_DIR`). A number that the main text no longer states in the checked wording but the supplement does is reported as moved, with its value still checked. Its exit status is 0 if every check passes, 1 if any check fails and 2 if either text is missing, in which case no check runs. `ISIC_META` may point at the ISIC-2024 `train-metadata.csv`. `verify_paper3.py` is the verifier of an earlier, longer version of the manuscript, which is not included; it is kept for the record.

## Pre-specified route test on held-out images (c62 to c65)

`code/c62_route_test.py` asks whether exact routes can be built on held-out images under a footprint made of the four magnitude descriptors (altered-pixel fraction, mean absolute change, spatial support, lesion coverage), and whether the route sums of different controls agree on the audited EfficientNet-B3 classifiers. Every control keeps the concept edit's per-pixel change magnitude, so it matches that footprint exactly. The controls form a two-by-two design: change direction (radial, hue) by patch scale (4, 16 px).

The plan is in `results/c62_preregistration.json`. It fixes the locked image set (every malignant image of the selection fold: 78 images from 54 patients), the gates, the analysis sets, the score floor, the 0.5-logit margin and the decision rule. It was committed with the script in commit 40b6367, before any response to an edited image of the locked set was computed; The commit was first pushed to the repository [Causal_Audits_of_SLC_v2](https://github.com/dqtoan87/Causal_Audits_of_SLC_v2), whose GitHub activity log records the push at 02:21:30 UTC on 2026-11-15; this repository carries the same history. It is a locked plan in this repository, not an entry in an external registry. The evaluation phase stops if `code/c62_route_test.py` differs from the SHA-256 recorded in the plan.

```bash
PY=/path/to/python bash code/run_c62.sh   # construct (no model), then evaluate and frontier; logs in log/c62_*.log
```

| Script | Output in `results/` | Computes |
| --- | --- | --- |
| `c62_route_test.py` | `c62_preregistration.json`, `c62_construction.json`, `c62_evaluation.json`, `c62_frontier.json` | the pre-specified test: construction without a model, cycle test on the audited classifiers, six-descriptor frontier |
| `c63_route_exploratory.py` | `c63_route_exploratory.json`, `c63_route_scores.npz` | exploratory: each pair restricted to images on which both controls pass the independent check; stores every per-image response |
| `c64_route_screens.py` | `c64_route_screens.json` | exploratory and model-free: the observable screens under the six-descriptor footprint, and off-target concept changes |
| `c65_route_robustness.py` | `c65_route_robustness.json` | exploratory, from the stored responses: the margins at which verdicts change, other bootstrap seeds, leave one patient out, the two cross pairs |
| `c67_route_sensitivity.py` | `c67_route_sensitivity.json` | exploratory: the score floor applied per model or fixed, full precision, mixed batches, saturation at the clipping bound |
| `c69_route_clip_sensitivity.py` | `c69_route_clip_sensitivity.json` | exploratory: clip levels 1e-4 to 1e-8 read from the head's log-odds in bf16 and fp32, the floor read in fp32, the probability scale, the largest log-odds |
| `c73_route_robustness_table.py` | `c73_route_robustness_table.json` | the decisive differences under the post hoc variants of the plan's analysis, and the robust cells of each run |
| `c74_route_exact_match.py` | `c74_route_exact_match.json` | exploratory, from the stored responses: the cycle test on the images where both controls of a difference match the four magnitude descriptors bit for bit, with the plan's 30-image rule and with all 48 differences |
| `c76_route_exact_detail.py` | `c76_route_exact_detail.json` | exploratory, from the stored responses: each decisive difference under the gate and the two bit-exact families, the exact identity on bit-exact image sets, and leave one patient out for the bit-exact Inv decision |
| `c77_route_feature_shift.py` | `c77_route_feature_shift.json` | exploratory and model-free: how far edits and controls move images in an independent ImageNet ResNet-50 feature space |
| `c78_batch_context.py` | `c78_batch_context.json` | exploratory: the route-test images rescored one image per batch (bf16 and fp32) and in mixed batches, to measure how much the batch context moves each score and the cycle test |
| `c79_route_near_natural.py` | `c79_route_near_natural.json` | exploratory: the cycle test restricted to images on which both controls stay within the natural 95th percentile of the independent feature space |
| `c80_route_null_calibration.py` | `c80_route_null_calibration.json`, `c80_route_null_draws_first.npz`, `c80_route_null_draws_second.npz` | exploratory: size and power of the cycle-test band by a double bootstrap with the analysed patients as the population, and the band recalibrated to 95% joint coverage |
| `c81_protocol_checks.py` | `c81_protocol_checks.json` | the inference protocol of the concept library and the route test: one batch list per image, and identical responses for the duplicate-label operator pairs |
| `c82_image_set_checks.py` | `c82_image_set_checks.json`, `c82_library83_cache.npz` | post hoc, model-free: the exact graph on the 83 malignant images where the library responses are scored (library rebuilt exactly as scored), and the Lipschitz floor and break-even table on one image set and one protocol (58 images, library protocol) |
| `c83_batch_extremes.py` | `c83_batch_extremes.json`, `c83_batch_extremes_scores.npz` | exploratory: where the large batch- and precision-dependent score changes of the route test come from, and the cycle test under four protocols |
| `c86_batch_context_first_run.py` | `c86_batch_context_first_run.json` | exploratory: the cycle test of the first run (ERM, Slim and the epoch-1 Inv checkpoint of `checkpoints-v1.0`) under the same four batch and precision protocols |
| `c87_route_feature_strata.py` | `c87_route_feature_strata.json` | exploratory, stored outputs only: the differences that decide the route-test flags, split by whether the controls drive the pooled backbone feature beyond ten times its usual norm |
| `run_retrain_b3inv.sh` | `c62_evaluation_inv_retrained.json`, `c63_route_exploratory_inv_retrained.json`, `c63_route_scores_inv_retrained.npz`, `c65_route_robustness_inv_retrained.json` | the second run: the unchanged plan, script and construction applied to the trio with the EfficientNet-B3 Inv checkpoint of `checkpoints-v1.1` |

`c63` scores the locked images again; `c64` and `c65` read no new model output.

**Results.** The plan's procedure was run twice without deviation; the second run applied it to the retrained Inv checkpoint. Every control matched the magnitude footprint to numerical precision (one pixel in the altered-pixel fraction, a relative 1e-5 elsewhere), and all four controls passed the exactness and neutrality gate on 74 to 78 of the 78 images per target. The concept edit's response cancels from every difference of two route sums, so the cycle test compares the controls with each other. The decision uses a simultaneous 95% band over 48 differences and a 0.5-logit margin.

- First run (plan-locked outcome; EfficientNet-B3 Inv checkpoint of `checkpoints-v1.0`; files without suffix): on the primary set (73 to 77 images, 52 to 54 patients) the plan's nominal band flags 3 of 12 model-by-target cells as disagreeing (ERM and Slim, all through the radial scale pair), agree in 2 (the Inv model at both low doses) and are indeterminate in 7. The two highest-dose disagreements are robust to the bootstrap seed and to leaving out any patient; the third (Slim, lowest contrast dose) is borderline.
- Second run (the unchanged plan applied after the Inv checkpoint was retrained; files with the suffix `_inv_retrained`): the score floor reads all three models, so the primary set changes to 44 to 47 images from 32 or 33 patients. The nominal band flags 3 of 12 cells, one per classifier (ERM and Inv at the highest contrast dose, Slim at the highest variegation dose), none agree and 9 are indeterminate. The ERM and Inv disagreements are robust to the bootstrap seed and to leaving out any patient; the Slim one depends on single patients.
- Across the post hoc variants of the plan's analysis (seeds, leave one patient out, score floor, full precision, mixed batches, one image per batch, clip levels 1e-4 to 1e-8; `c65`, `c67`, `c69`, `c73`, `c83`, `c86`), each applied to both runs one element at a time, two of the three disagreements of each run hold. Only ERM at the highest contrast dose holds in both runs. Every decision is made on the clipped logit scale; the unclipped log-odds are not stable across numerics.
- On images whose footprints match bit for bit (`c74`), the decisive differences keep their size. With all 48 differences, one cell still disagrees in the second run (Inv) and two in the first (ERM and Slim), so which cells disagree, and how many, depend on the analysis; in every analysis at least one cell per run disagrees.
- Read exactly, on the images where both controls match bit for bit, none of the 48 differences is zero in either run (`c76`), so C1a for the magnitude footprint and control neutrality cannot both hold exactly in any cell; this is a weak statement, since exact equality of floating-point outputs on distinct images is not expected, and the 0.5-logit test adds which differences are large. Under the 30-image rule the bit-exact reanalysis gives four and three disagreeing cells. The bit-exact Inv decision holds in 26 of 27 leave-one-patient-out subsets.
- Per overlay realization (`c75`), the joint frontier runs from 0.041 to 0.056. In an independent ImageNet encoder (`c77`) the controls show no feature blow-up, but at the high doses they move images further than the concept edit does.
- The first run is the plan-locked analysis; the second is a non-independent reapplication after the checkpoint correction. Scoring each image alone (`c78`) moves scores by a median of 0.016 logit; ERM and Inv at the highest contrast dose disagree under every batch protocol, the Slim decisions move. On near-natural images (`c79`) the ERM and Inv pairs keep 16 images and are not decided.
- Under a known null (`c80`), the plan's band covers the 48 differences jointly in 77.7% (first run) and 73.2% (second run) of resampled cohorts, not 95%, although each difference's own interval covers in at least 95.7%. Widening every half-width by 1.42 and 1.47 restores 95% joint coverage; with that band only ERM at the highest contrast dose in the first run still disagrees, and no cell of the second run does. The plan-locked counts are therefore nominal counts on the pre-specified band; the post hoc recalibration qualifies them.
- Batch context and precision (`c83`): every network runs in evaluation mode with no batch-statistic layer, and the plan's protocol reproduces the stored scores exactly. Of the scores that move by more than 0.5 logit when scored one image per batch, 86% come from inputs whose pooled feature exceeds ten times its usual norm (median 524 times); other scores move that much in 0.3% of cases. In the second run, ERM and Inv at the highest contrast dose disagree under all four combinations of batch context and precision; in the first run (`c86`), ERM at the highest contrast dose and Slim at the highest variegation dose do, with smallest lower bounds of 1.54 logits.
- Feature range (`c87`): on images where either control drives the pooled backbone feature beyond ten times its median unedited norm, the decisive differences are +8.2 to +17.8 logits and carry 75 to 92% of each high-dose flag; on the other images they are +0.8 to +1.9 logits, and six of seven marginal intervals stay above the 0.5-logit margin.
- Image sets (`c82`): on the 83 malignant images, rebuilt exactly as scored, the exact graph has three more duplicate-label pairs (one control family at the same strength) and stays disconnected. On one set and one protocol (58 images), the Lipschitz floor is L^g >= 1.09 and the break-even concept-side constant is 0.152 to 0.186 at every target.

A disagreement on the plan's band is a nominal flag: the band undercovers jointly (`c80`), so its nominal family-wise level is not attained. Reading gate matches as equal footprints, a flag points against the magnitude-only working form jointly with control neutrality; it does not test the six-descriptor footprint. Agreement would certify no route sum, because a bias shared by all routes cancels, and any route also needs its controls to be latently neutral, which no footprint shows. Under the six-descriptor footprint and the library's screens, no constructed control passes all three screens.

## License

MIT, see `LICENSE`.
