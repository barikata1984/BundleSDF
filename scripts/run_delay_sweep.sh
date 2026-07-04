#!/usr/bin/env bash
# sync_max_delay 精度スイープ. 計画: notes/PLAN_delay_accuracy_sweep.md
#
# 各 (video, delay) について config.yml の sync_max_delay を書き換えて
# run_ho3d.py -> benchmark_ho3d.py を逐次実行し, 結果を
#   data/bench_results/ho3d_sweep/<video>_delay<N>/
#     ho3d_ours/<video>/ob_in_cam/*.txt   (姿勢出力)
#     ho3d_log/ho3d_ours.pkl              (ADD/ADD-S/AUC/chamfer)
# に保存する.
#
# set -e は使わない: 8 時間の無人実行で 1 回の失敗が全体を殺すのを避けるため,
# 各コマンドの exit code を明示的に確認し, 失敗は記録して次の組へ進む
# (「完了を待ってから次へ」という要件は満たしつつ, 予算を無駄にしない).

set -u -o pipefail

REPO=/workspace
cd "$REPO"

DELAYS=(3 4 5 6 7 8 10 15)
VIDEOS=(AP10 MPM10 SB11)
SWEEP_ROOT="$REPO/data/bench_results/ho3d_sweep"
CONFIG="$REPO/config.yml"
BUDGET_SECONDS=$((8 * 3600))
SCRIPT_START=$(date +%s)

MPS_DIR=/tmp/nvidia-mps-sweep
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

set_delay() { sed -i -E "s/^(sync_max_delay:[[:space:]]*)[0-9]+/\1$1/" "$CONFIG"; }

cleanup() {
  log "cleanup: restore sync_max_delay=3, stop MPS"
  set_delay 3
  echo quit | nvidia-cuda-mps-control 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$SWEEP_ROOT"

# --- MPS 起動 (専用パイプディレクトリ) ---
rm -rf "$MPS_DIR"
mkdir -p "$MPS_DIR"
log "starting MPS control daemon (pipe dir=$MPS_DIR)"
nvidia-cuda-mps-control -d || log "WARN: nvidia-cuda-mps-control -d returned nonzero"
sleep 2
if echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null 2>&1; then
  log "MPS daemon responding"
else
  log "WARN: MPS daemon not responding; runs will proceed without MPS"
fi

run_count=0
total_run_seconds=0
declare -a FAILURES=()

run_one() {
  local video=$1 delay=$2
  local tag="${video}_delay${delay}"
  local tag_dir="$SWEEP_ROOT/$tag"
  local out_dir="$tag_dir/ho3d_ours"
  local log_dir="$tag_dir/ho3d_log"
  local vdir="$REPO/data/HO3D_v3/evaluation/$video"
  mkdir -p "$out_dir" "$log_dir"

  log "START $tag : set sync_max_delay=$delay"
  set_delay "$delay"

  local t0 t1 elapsed nposes
  t0=$(date +%s)

  log "$tag : run_ho3d.py (log -> $tag_dir/run.log)"
  if ! python3 run_ho3d.py --video_dirs "$vdir" --out_dir "$out_dir" >"$tag_dir/run.log" 2>&1; then
    log "FAIL $tag : run_ho3d.py nonzero exit; skip benchmark"
    FAILURES+=("$tag:run")
    return 1
  fi
  nposes=$(ls "$out_dir/$video/ob_in_cam/"*.txt 2>/dev/null | wc -l)
  log "$tag : run_ho3d done, $nposes pose files"

  log "$tag : benchmark_ho3d.py (log -> $tag_dir/benchmark.log)"
  if ! python3 benchmark_ho3d.py --video_dirs "$vdir" --out_dir "$out_dir" --log_dir "$log_dir" >"$tag_dir/benchmark.log" 2>&1; then
    log "FAIL $tag : benchmark_ho3d.py nonzero exit"
    FAILURES+=("$tag:bench")
    return 1
  fi

  t1=$(date +%s)
  elapsed=$((t1 - t0))
  run_count=$((run_count + 1))
  total_run_seconds=$((total_run_seconds + elapsed))
  # ADD/ADD-S 等の要約行を benchmark ログから拾って進捗に出す
  grep -E "video ${video}, ADD" "$tag_dir/benchmark.log" | tail -1 | sed "s/^/    [${tag}] /" || true
  log "DONE $tag : elapsed=${elapsed}s (runs=$run_count, total_run=${total_run_seconds}s)"
}

log "=== delay sweep start: videos=(${VIDEOS[*]}) delays=(${DELAYS[*]}) budget=${BUDGET_SECONDS}s ==="

for video in "${VIDEOS[@]}"; do
  # SB11 (3 本目) は予算残を見て実行判断する (計画のバッファ運用)
  if [[ "$video" == "SB11" ]]; then
    now=$(date +%s)
    spent=$((now - SCRIPT_START))
    if [[ $run_count -gt 0 ]]; then
      avg=$((total_run_seconds / run_count))
    else
      avg=900
    fi
    projected=$((spent + avg * ${#DELAYS[@]}))
    log "budget check before SB11: spent=${spent}s avg_per_run=${avg}s projected_total=${projected}s budget=${BUDGET_SECONDS}s"
    if [[ $projected -gt $BUDGET_SECONDS ]]; then
      log "SKIP SB11: projected_total ${projected}s > budget ${BUDGET_SECONDS}s"
      continue
    fi
    log "PROCEED SB11: within budget"
  fi

  for delay in "${DELAYS[@]}"; do
    run_one "$video" "$delay" || true
  done
done

log "=== sweep finished: total runs=$run_count, total_run_time=${total_run_seconds}s, wall=$(( $(date +%s) - SCRIPT_START ))s ==="
if [[ ${#FAILURES[@]} -gt 0 ]]; then
  log "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
else
  log "no failures"
fi
