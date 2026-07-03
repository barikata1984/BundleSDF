#!/bin/bash
# Online tracking benchmark for the BundleSDF milk-carton demo, run inside the
# bundlesdf:ros-one container (e.g. `docker exec -it bundlesdf_bench bash`).
#
# For each matcher backend (eloftr, loftr) this:
#   1) runs scripts/bench_milk.py, which replays run_custom.py's online
#      tracking loop (tracker.run() per frame) and records per-frame timing
#      to a CSV plus a mean/median/p90 summary and frames/s (see that script
#      for why the offline global-refine/mesh step is intentionally skipped).
#   2) wraps the whole run with /usr/bin/time for a coarse end-to-end
#      wall-clock/RSS reading in addition to the per-frame CSV.
#
# use_segmenter is always 0: use_segmenter=1 has a known mask-resize mismatch
# bug (see notes/ISSUES.md), so this benchmark relies on the dataset-provided
# masks under video_dir/masks/ instead of online segmentation.
#
# After both backends have run, scripts/compare_poses.py compares the two
# resulting ob_in_cam pose trajectories as a relative consistency check (there
# is no ground truth for this demo video, so this is NOT an absolute
# ADD/ADD-S accuracy number -- see that script's docstring, and use
# benchmark_ho3d.py on the HO3D dataset for GT-based absolute error).
#
# Usage (run inside the container, working directory /workspace):
#   bash scripts/bench_milk.sh
# Optional overrides via environment variables:
#   VIDEO_DIR, OUT_ROOT, RESULTS_DIR, DEBUG_LEVEL, STRIDE, WARMUP_FRAMES

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VIDEO_DIR="${VIDEO_DIR:-/workspace/data/2022-11-18-15-10-24_milk}"
OUT_ROOT="${OUT_ROOT:-/workspace/data}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/data/bench_results}"
DEBUG_LEVEL="${DEBUG_LEVEL:-1}"
STRIDE="${STRIDE:-1}"
WARMUP_FRAMES="${WARMUP_FRAMES:-5}"

mkdir -p "${RESULTS_DIR}"

for BACKEND in eloftr loftr; do
  OUT_FOLDER="${OUT_ROOT}/out_milk_${BACKEND}"
  CSV_OUT="${RESULTS_DIR}/${BACKEND}_frame_times.csv"
  TIME_LOG="${RESULTS_DIR}/${BACKEND}_time.log"

  echo "=== [bench_milk] backend=${BACKEND} out_folder=${OUT_FOLDER} ==="

  BUNDLESDF_PROFILE=1 BUNDLESDF_MATCHER="${BACKEND}" /usr/bin/time -v -o "${TIME_LOG}" \
    python3 "${REPO_DIR}/scripts/bench_milk.py" \
      --video_dir "${VIDEO_DIR}" \
      --out_folder "${OUT_FOLDER}" \
      --use_segmenter 0 \
      --debug_level "${DEBUG_LEVEL}" \
      --stride "${STRIDE}" \
      --warmup_frames "${WARMUP_FRAMES}" \
      --csv_out "${CSV_OUT}" \
    2>&1 | tee "${RESULTS_DIR}/${BACKEND}_run.log"

  echo "=== [bench_milk] backend=${BACKEND} done; wall-clock/RSS: ${TIME_LOG} ==="
done

echo "=== [bench_milk] comparing eloftr vs loftr pose trajectories ==="
python3 "${REPO_DIR}/scripts/compare_poses.py" \
  --out_folder_a "${OUT_ROOT}/out_milk_eloftr" \
  --out_folder_b "${OUT_ROOT}/out_milk_loftr" \
  --csv_out "${RESULTS_DIR}/eloftr_vs_loftr_pose_diff.csv" \
  2>&1 | tee "${RESULTS_DIR}/compare_poses.log"

echo "=== [bench_milk] all results under: ${RESULTS_DIR} ==="
