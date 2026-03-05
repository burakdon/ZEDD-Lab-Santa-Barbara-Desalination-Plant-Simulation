#!/usr/bin/env bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DROUGHT="pers87_sev0.83n_4"
CASES=(
  "basetariff_baseline/3mpd_30vessels"
  "basetariff_baseline/3mpd_36vessels"
  "basetariff_baseline/4mpd_30vessels"
  "basetariff_baseline/4mpd_36vessels"
  "basetariff_baseline/6mpd_36vessels"
  "basetariff_baseline/8mpd_36vessels"
  "basetariff_flexible/3mpd_30vessels"
  "basetariff_flexible/3mpd_36vessels"
  "basetariff_flexible/4mpd_30vessels"
  "basetariff_flexible/4mpd_36vessels"
  "basetariff_flexible/6mpd_36vessels"
  "basetariff_flexible/8mpd_36vessels"
)

run_set () {
  local SET_NAME="$1"        # "new_data" or "old_data"
  local OUT_FIXED="$2"       # e.g., "result/fixed_desal_new"
  local ARCHIVE_DIR="$3"     # e.g., "result_newdata"

  echo "=== Running set: $SET_NAME ==="
  export COST_CURVES_SET="$SET_NAME"

  # FLEX runs
  python3 run_all_cases.py \
    --drought "$DROUGHT" \
    --cases "${CASES[@]}"

  # FIXED runs
  python3 fixed_desal_experiment/run_all_cases_fixed.py \
    --drought "$DROUGHT" \
    --cases "${CASES[@]}" \
    --outdir "$OUT_FIXED"

  # Archive full result tree
  rm -rf "$ARCHIVE_DIR"
  cp -r result "$ARCHIVE_DIR"
}

case "$1" in
  new)
    run_set "new_data" "result/fixed_desal_new" "result_newdata"
    ;;
  old)
    run_set "old_data" "result/fixed_desal_old" "result_olddata"
    ;;
  both|"")
    run_set "new_data" "result/fixed_desal_new" "result_newdata"
    run_set "old_data" "result/fixed_desal_old" "result_olddata"
    ;;
  *)
    echo "Usage: $0 [new|old|both]"
    exit 1
    ;;
esac

