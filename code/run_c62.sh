#!/usr/bin/env bash
# Runs the pre-registered route test (c62) end to end on the locked set:
#   construct (no model) -> evaluate (audited checkpoints, GPU) and frontier (no model, CPU) in parallel.
# The evaluation refuses to run if c62_route_test.py differs from the registered version.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; BASE="$(dirname "$HERE")"; LOG="$BASE/log"
PY="${PY:-python3}"
mkdir -p "$LOG"
echo "c62 start $(date -Iseconds)" > "$LOG/c62_run.log"
"$PY" "$HERE/c62_route_test.py" --phase construct > "$LOG/c62_construct.log" 2>&1 || { echo "construct failed" >> "$LOG/c62_run.log"; exit 1; }
echo "construct done $(date -Iseconds)" >> "$LOG/c62_run.log"
"$PY" "$HERE/c62_route_test.py" --phase evaluate > "$LOG/c62_evaluate.log" 2>&1 &
EV=$!
"$PY" "$HERE/c62_route_test.py" --phase frontier > "$LOG/c62_frontier.log" 2>&1
echo "frontier exit $? $(date -Iseconds)" >> "$LOG/c62_run.log"
wait $EV
echo "evaluate exit $? $(date -Iseconds)" >> "$LOG/c62_run.log"
echo "c62 done $(date -Iseconds)" >> "$LOG/c62_run.log"
