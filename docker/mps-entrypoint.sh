#!/usr/bin/env bash
# Start the CUDA MPS control daemon, then run CMD. On SIGTERM/SIGINT the daemon
# is asked to quit gracefully. exec is intentionally avoided: it would replace
# this shell and discard the trap, leaving the daemon un-quit on `docker stop`.

: "${CUDA_MPS_PIPE_DIRECTORY:=/run/nvidia-mps}"
: "${CUDA_MPS_LOG_DIRECTORY:=/run/nvidia-mps}"
export CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Non-fatal: if MPS cannot start, still bring the container up.
nvidia-cuda-mps-control -d || echo "mps-entrypoint: failed to start MPS control daemon" >&2

"$@" &
child=$!

shutdown() {
    trap - SIGTERM SIGINT
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
}
trap shutdown SIGTERM SIGINT

wait "$child"
