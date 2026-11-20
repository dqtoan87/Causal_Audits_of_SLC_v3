#!/bin/bash
# Retrain the EfficientNet-B3 Inv checkpoint and re-run every analysis that reads it.
#
# Why. train_causal kept the best selection-fold AUROC over all epochs, the three warm-up epochs
# included. The EfficientNet-B3 Inv model peaked at its first epoch, so the saved weights never
# carried the penalties. Selection now reads only epochs with both penalties on
# (CFG3 select_after_warmup, set by c29). The ResNet-50 Inv model was saved at epoch 6 under
# either rule, so it is not retrained.
#
# Steps
#   1. c29, job dj_causal_b3 only               -> checkpoints/dj_causal_b3.pt
#      (the previous file is kept in checkpoints/archived_b3inv_ep1/)
#   2. c44                                       -> results/c19_preds.npz, read by c26, c32, c38 and
#                                                   the shrinkage figure (run_all.sh does not run c44)
#   3. the analyses of run_all.sh, except c35, which trains its own models and reads no audited checkpoint;
#      c51, c58 and c59 run only when OTHER points at the reimplementation, as in run_all.sh
#   4. the route test under its unchanged plan: c62 --phase evaluate, then c63 and c65. Their outputs
#      are renamed *_inv_retrained.* and the registered files are restored byte for byte, so the
#      registered outcome stays on record (checked against results/c62_manifest.sha256).
#   5. verify_paper3.py (mismatches are expected until the text is updated) and a field-level diff
#      against results/_pre_b3inv_retrain.
#
# Preconditions, created before launch: checkpoints/archived_b3inv_ep1/dj_causal_b3.pt,
# results/_pre_b3inv_retrain/ and log/_pre_b3inv_retrain/.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
LOG="$BASE/log"; RES="$BASE/results"; CK="$BASE/checkpoints"
PY="${PY:-python3}"
OTHER="${OTHER:-}"   # root of the separate reimplementation; c51, c58 and c59 run only when it is set
BK="$RES/_pre_b3inv_retrain"
DIFF_TOOL="${DIFF_TOOL:-}"
M="$LOG/run_retrain_b3inv.log"
ROUTE_FILES=(c62_evaluation.json c63_route_exploratory.json c63_route_scores.npz c65_route_robustness.json)

stamp() { date +%H:%M:%S; }
note() { echo "[$(stamp)] $*" >> "$M"; }

echo "=== retrain B3 Inv started $(date -Is) ===" > "$M"
rm -f "$LOG/run_retrain_b3inv.done"

# ---------------------------------------------------------------- preconditions
[ -f "$CK/archived_b3inv_ep1/dj_causal_b3.pt" ] || { note "missing archived checkpoint"; exit 1; }
for f in "${ROUTE_FILES[@]}"; do
    [ -f "$BK/$f" ] || { note "missing backup $BK/$f"; exit 1; }
done
note "old checkpoint $(sha256sum "$CK/archived_b3inv_ep1/dj_causal_b3.pt" | cut -c1-16)"

# ---------------------------------------------------------------- 1. training
note "step 1: c29 dj_causal_b3 (selection after warm-up)"
C29_ONLY=dj_causal_b3 C29_FORCE=1 "$PY" "$HERE/c29_disjoint_train.py" > "$LOG/c29_b3inv_retrain.log" 2>&1
rc=$?
grep -E "dj_causal_b3\] (ep|early|saved)" "$LOG/c29_b3inv_retrain.log" | sed 's/^/      | /' >> "$M"
if [ $rc -ne 0 ]; then note "training failed rc=$rc"; tail -n 20 "$LOG/c29_b3inv_retrain.log" | sed 's/^/      | /' >> "$M"; exit 1; fi
NEW=$(sha256sum "$CK/dj_causal_b3.pt" | cut -c1-64); OLD=$(sha256sum "$CK/archived_b3inv_ep1/dj_causal_b3.pt" | cut -c1-64)
[ "$NEW" != "$OLD" ] || { note "checkpoint unchanged after training; stopping"; exit 1; }
note "new checkpoint ${NEW:0:16}"

# ---------------------------------------------------------------- 2. prediction cache
note "step 2: c44 prediction cache"
"$PY" "$HERE/c44_cache_preds.py" > "$LOG/c44_cache_preds.log" 2>&1 || { note "c44 failed"; tail -n 15 "$LOG/c44_cache_preds.log" | sed 's/^/      | /' >> "$M"; exit 1; }

# ---------------------------------------------------------------- 3. analyses
SCRIPTS=(
  c54_synthetic_scm c2_cace_artifact c3_cace_concept c7_extra_controls c10_multiplacebo
  c12_matched_control c13_fullpop c19_saturation_mde c20_numerical_repro c21_patient_cluster
  c25_screen_full_sample c26_hair_dose_prob c30_disjoint_audit c32_absdp_patient_ci c34_ctrl_vs_ctrl
  c38_shrinkage_sensitivity c40_simultaneous c56_simultaneous36 c57_stratified_band c60_design_coverage
  c42_positive_control c45_partial_id c46_padufes_artifact c49_tcav c50_overlay_seed
  c51_independent_rebuild c58_rebuild_trace c59_rebuild_generator c52_overlay_perceptual
  c55_identification_graph make_figures_split make_fig_shrinkage
)
args_for() {
    case "$1" in
        c46_padufes_artifact) echo "--use-saved" ;;
        c51_independent_rebuild|c58_rebuild_trace|c59_rebuild_generator) echo "--other $OTHER" ;;
    esac
}
ok=0; fail=0; failed=()
note "step 3: ${#SCRIPTS[@]} analyses"
for s in "${SCRIPTS[@]}"; do
    t0=$(date +%s)
    # shellcheck disable=SC2046
    if { [ "$s" = c51_independent_rebuild ] || [ "$s" = c58_rebuild_trace ] || [ "$s" = c59_rebuild_generator ]; } && [ -z "$OTHER" ]; then
        note "$(printf '%-26s skipped (set OTHER)' "$s")"; continue
    fi
    "$PY" "$HERE/$s.py" $(args_for "$s") > "$LOG/$s.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); note "$(printf '%-26s ok   %5ds' "$s" "$(( $(date +%s)-t0 ))")"
    else fail=$((fail+1)); failed+=("$s"); note "$(printf '%-26s FAIL rc=%d %5ds' "$s" "$rc" "$(( $(date +%s)-t0 ))")"
         tail -n 12 "$LOG/$s.log" | sed 's/^/      | /' >> "$M"; fi
done
note "analyses ok=$ok fail=$fail ${failed[*]:-}"

# ---------------------------------------------------------------- 4. route test, unchanged plan
note "step 4: route test with the retrained Inv model (registered files restored afterwards)"
"$PY" "$HERE/c62_route_test.py" --phase evaluate > "$LOG/c62_evaluate_inv_retrained.log" 2>&1; r62=$?
r63=9; r65=9
if [ $r62 -eq 0 ]; then
    "$PY" "$HERE/c63_route_exploratory.py" > "$LOG/c63_route_exploratory_inv_retrained.log" 2>&1; r63=$?
fi
if [ $r63 -eq 0 ]; then   # c65 reads the c63 scores, so it runs only on a fresh c63
    "$PY" "$HERE/c65_route_robustness.py" > "$LOG/c65_route_robustness_inv_retrained.log" 2>&1; r65=$?
fi
note "c62 rc=$r62  c63 rc=$r63  c65 rc=$r65"
rc_for() { case "$1" in c62_*) echo $r62 ;; c63_*) echo $r63 ;; c65_*) echo $r65 ;; esac; }
for f in "${ROUTE_FILES[@]}"; do
    b="${f%.*}"; e="${f##*.}"
    if ! cmp -s "$RES/$f" "$BK/$f"; then
        if [ "$(rc_for "$f")" -eq 0 ]; then mv "$RES/$f" "$RES/${b}_inv_retrained.$e"
        else mv "$RES/$f" "$RES/${b}_inv_retrained_FAILED.$e"; fi
    fi
    cp -p "$BK/$f" "$RES/$f"
done
( cd "$BASE" && sha256sum -c results/c62_manifest.sha256 ) >> "$M" 2>&1 && note "registered route-test files intact" \
    || note "WARNING: route-test manifest check failed"

# ---------------------------------------------------------------- 5. verification and diff
"$PY" "$HERE/verify_paper3.py" > "$LOG/verify_paper3_after_b3inv.log" 2>&1
note "verifier: $(tail -n 1 "$LOG/verify_paper3_after_b3inv.log")"
if [ -n "$DIFF_TOOL" ] && [ -f "$DIFF_TOOL" ]; then
    "$PY" "$DIFF_TOOL" "$RES" "$BK" "$LOG/b3inv_retrain_diff.txt" >> "$M" 2>&1
fi
echo "=== finished $(date -Is) ===" >> "$M"
date -Is > "$LOG/run_retrain_b3inv.done"
