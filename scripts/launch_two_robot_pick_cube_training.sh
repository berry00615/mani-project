#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 METHOD PHYSICAL_GPU RUN_NAME LOG_NAME" >&2
  exit 2
fi

METHOD=$1
PHYSICAL_GPU=$2
RUN_NAME=$3
LOG_NAME=$4

WRITABLE_ROOT=/vepfs-mlp2/queue013/public/huangborui
PROJECT_ROOT=$WRITABLE_ROOT/mani-project
PYTHON_BIN=$WRITABLE_ROOT/envs/stackcube_official/bin/python
TRAINER=$PROJECT_ROOT/third_party/ManiSkill-v3.0.1-official/ppo/ppo_fast.py
LOG_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/$LOG_NAME
RUN_DIR=$PROJECT_ROOT/runs/$RUN_NAME
CACHE_DIR=$WRITABLE_ROOT/runtime_cache
TEMP_DIR=$WRITABLE_ROOT/runtime_tmp

case "$(realpath -m "$LOG_DIR")/" in
  "$WRITABLE_ROOT"/*) ;;
  *) echo "log path escapes server write boundary" >&2; exit 91 ;;
esac
case "$(realpath -m "$RUN_DIR")/" in
  "$WRITABLE_ROOT"/*) ;;
  *) echo "run path escapes server write boundary" >&2; exit 91 ;;
esac
[[ ! -e "$LOG_DIR" ]] || { echo "refusing existing log directory" >&2; exit 92; }
[[ ! -e "$RUN_DIR" ]] || { echo "refusing existing run directory" >&2; exit 92; }
[[ "$PHYSICAL_GPU" == 0 || "$PHYSICAL_GPU" == 1 ]]

case "$METHOD" in
  official)
    CONFIG=$PROJECT_ROOT/configs/two_robot_pick_cube/official_ppo_fast_50m.yaml
    EXTRA_ARGS=()
    ;;
  long_credit)
    CONFIG=$PROJECT_ROOT/configs/two_robot_pick_cube/long_credit_ppo_fast_50m.yaml
    EXTRA_ARGS=(--gamma 0.99 --gae-lambda 0.95)
    ;;
  moderate_credit)
    CONFIG=$PROJECT_ROOT/configs/two_robot_pick_cube/moderate_credit_ppo_fast_50m.yaml
    EXTRA_ARGS=(--gamma 0.95 --gae-lambda 0.95)
    ;;
  *)
    echo "unknown method: $METHOD" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR" "$CACHE_DIR" "$TEMP_DIR"
cp "$CONFIG" "$LOG_DIR/config.yaml"

COMMAND=(
  "$PYTHON_BIN" "$TRAINER"
  --env-id TwoRobotPickCube-v1
  --seed 0
  --num-envs 1024
  --num-steps 100
  --update-epochs 8
  --num-minibatches 32
  --total-timesteps 50000000
  --num-eval-envs 16
  --num-eval-steps 100
  --cudagraphs
  --exp-name "$RUN_NAME"
  "${EXTRA_ARGS[@]}"
)

{
  echo "experiment_method=$METHOD"
  echo "physical_gpu=$PHYSICAL_GPU"
  echo "cuda_visible_device=0"
  echo "launch_topology=one independent PPO-fast process on one physical GPU"
  echo "run_name=$RUN_NAME"
  echo "log_name=$LOG_NAME"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "working_directory=$PROJECT_ROOT"
  echo "python=$PYTHON_BIN"
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c \
    'import torch, mani_skill, gymnasium, sapien; print("torch="+torch.__version__); print("mani_skill="+mani_skill.__version__); print("gymnasium="+gymnasium.__version__); print("sapien="+sapien.__version__)'
  sha256sum "$TRAINER" "$CONFIG"
  printf "command="
  printf "%q " "${COMMAND[@]}"
  printf "\n"
} >"$LOG_DIR/launch_manifest.txt"

nvidia-smi -i "$PHYSICAL_GPU" -q >"$LOG_DIR/gpu_before.txt"
cd "$PROJECT_ROOT"
export XDG_CACHE_HOME=$CACHE_DIR
export TMPDIR=$TEMP_DIR
export PYTHONUNBUFFERED=1

set +e
CUDA_VISIBLE_DEVICES=$PHYSICAL_GPU "${COMMAND[@]}" \
  > >(tee "$LOG_DIR/console.log") \
  2> >(tee "$LOG_DIR/stderr.log" >&2) &
TRAIN_PID=$!
set -e
echo "$TRAIN_PID" >"$LOG_DIR/train.pid"
echo "$$" >"$LOG_DIR/launcher.pid"

(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    {
      echo "timestamp=$(date --iso-8601=seconds)"
      nvidia-smi -i "$PHYSICAL_GPU" \
        --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
      find "$RUN_DIR" -maxdepth 1 -type f -name '*.pt' -printf '%f %s bytes\n' \
        2>/dev/null | sort
    } >>"$LOG_DIR/monitor_180s.log"
    sleep 180
  done
) &
MONITOR_PID=$!
echo "$MONITOR_PID" >"$LOG_DIR/monitor.pid"

set +e
wait "$TRAIN_PID"
STATUS=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=$STATUS"
} >"$LOG_DIR/completion.txt"
nvidia-smi -i "$PHYSICAL_GPU" -q >"$LOG_DIR/gpu_after.txt"
find "$RUN_DIR" -maxdepth 1 -type f -printf '%f %s bytes\n' \
  2>/dev/null | sort >"$LOG_DIR/final_artifact_inventory.txt"
exit "$STATUS"
