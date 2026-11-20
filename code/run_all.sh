#!/bin/bash
# Independent re-run of every analysis the manuscript rests on.
#
# Everything resolves relative to this tree: the scripts derive their output directory as
# <parent of code/>/results, so nothing here can touch the results of any other tree.
#   results  -> ./results
#   logs     -> ./log        (one file per script, plus run_all.log)
#   models   -> ./checkpoints (own copies, not shared)
#
# Training is NOT part of this run. The six audited checkpoints are frozen and the paper's
# claims are about them, so re-running the analysis against those weights is what verifies
# the reported numbers. To retrain instead, run c29_disjoint_train.py and c8_backbone2_train.py
# first; they will overwrite ./checkpoints only, never the originals.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
LOG="$BASE/log"
RES="$BASE/results"
PY="${PY:-python3}"
MASTER="$LOG/run_all.log"

mkdir -p "$LOG" "$RES" "$BASE/paper/figures"

# analysis scripts, in dependency order (the prediction cache first; c25 before c45, c34 before c55)
SCRIPTS=(
  c44_cache_preds
  c54_synthetic_scm
  c2_cace_artifact
  c3_cace_concept
  c7_extra_controls
  c10_multiplacebo
  c12_matched_control
  c13_fullpop
  c19_saturation_mde
  c20_numerical_repro
  c21_patient_cluster
  c25_screen_full_sample
  c26_hair_dose_prob
  c30_disjoint_audit
  c32_absdp_patient_ci
  c34_ctrl_vs_ctrl
  c35_selection_isolation
  c38_shrinkage_sensitivity
  c40_simultaneous
  c56_simultaneous36
  c57_stratified_band
  c60_design_coverage
  c42_positive_control
  c45_partial_id
  c46_padufes_artifact
  c49_tcav
  c50_overlay_seed
  c51_independent_rebuild
  c58_rebuild_trace
  c59_rebuild_generator
  c52_overlay_perceptual
  c55_identification_graph
  # post hoc checks; c67 and c69 read the route-test results and need the first-run Inv checkpoint
  # in checkpoints/archived_b3inv_ep1 (release checkpoints-v1.0)
  c66_overlay_occlusion
  c70_occlusion_mask_sensitivity
  c68_floor_sets
  c71_rule_checks
  c72_library_proxy_split
  c67_route_sensitivity
  c69_route_clip_sensitivity
  c73_route_robustness_table
  c74_route_exact_match
  c75_seed_band
  c76_route_exact_detail
  c77_route_feature_shift
  c78_batch_context
  c79_route_near_natural
  c80_route_null_calibration
  c81_protocol_checks
  c82_image_set_checks
  c83_batch_extremes
  c84_label_provenance
  c85_artifact_band_calibration
  c86_batch_context_first_run
  c87_route_feature_strata
  c88_artifact_inv_epoch1
  make_figures_split
  make_fig_edit_examples
  make_fig_shrinkage
)

{
  echo "=== run started $(date -Is) ==="
  echo "tree      : $BASE"
  echo "python    : $PY"
  echo "scripts   : ${#SCRIPTS[@]}"
  "$PY" -c "import torch;print('torch     :',torch.__version__,'| cuda',torch.cuda.is_available())" 2>&1
  echo
} > "$MASTER"

# Per-script arguments. c46 audits the released PAD-UFES-20 checkpoints rather than retraining them,
# which is how the reported numbers were produced. c51, c58 and c59 compare against a separate
# reimplementation; they run only when OTHER points at its root directory.
args_for() {
    case "$1" in
        c46_padufes_artifact) echo "--use-saved" ;;
        c51_independent_rebuild|c58_rebuild_trace|c59_rebuild_generator) [ -n "${OTHER:-}" ] && echo "--other $OTHER" ;;
    esac
}

ok=0; fail=0; failed=(); skipped=()
for s in "${SCRIPTS[@]}"; do
    if { [ "$s" = c51_independent_rebuild ] || [ "$s" = c58_rebuild_trace ] || [ "$s" = c59_rebuild_generator ]; } && [ -z "${OTHER:-}" ]; then
        skipped+=("$s"); printf '%-26s skipped (set OTHER to the reimplementation root)\n' "$s" >> "$MASTER"
        continue
    fi
    t0=$(date +%s)
    printf '%-26s start %s\n' "$s" "$(date +%H:%M:%S)" >> "$MASTER"
    # shellcheck disable=SC2046
    "$PY" "$HERE/$s.py" $(args_for "$s") > "$LOG/$s.log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -eq 0 ]; then
        ok=$((ok+1));   printf '%-26s   ok  %5ds\n' "$s" "$((t1-t0))" >> "$MASTER"
    else
        fail=$((fail+1)); failed+=("$s")
        printf '%-26s FAIL rc=%d %5ds  (see log/%s.log)\n' "$s" "$rc" "$((t1-t0))" "$s" >> "$MASTER"
        tail -n 15 "$LOG/$s.log" | sed 's/^/      | /' >> "$MASTER"
    fi
done

{
  echo
  echo "=== run finished $(date -Is) ==="
  echo "ok=$ok fail=$fail"
  [ $fail -gt 0 ] && echo "failed: ${failed[*]}"
  [ ${#skipped[@]} -gt 0 ] && echo "skipped: ${skipped[*]}"
  echo "results written: $(ls -1 "$RES" | wc -l) files"
} >> "$MASTER"

# compare what was produced against the numbers the manuscript states
"$PY" "$HERE/verify_paper3.py" > "$LOG/verify_paper3.log" 2>&1
vrc=$?
case $vrc in
    0) echo "verification (paper3): passed, see log/verify_paper3.log" ;;
    2) echo "verification (paper3): NOT RUN, paper3.md or supplementary3.md missing in paper/ or PAPER_DIR (see log/verify_paper3.log)" ;;
    *) echo "verification (paper3): FAILED rc=$vrc, see log/verify_paper3.log" ;;
esac >> "$MASTER"
echo "=== all done $(date -Is) ===" >> "$MASTER"

# non-zero exit if any analysis failed or the verification did not pass
status=0
[ "$fail" -gt 0 ] && status=1
[ "$vrc" -ne 0 ] && status=1
echo "exit status $status" >> "$MASTER"
exit $status
