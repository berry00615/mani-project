#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 PHYSICAL_GPU LEARNING_RATE RUN_NAME LOG_NAME METHOD_ID" >&2
  exit 2
fi

PHYSICAL_GPU=$1
LEARNING_RATE=$2
RUN_NAME=$3
LOG_NAME=$4
METHOD_ID=$5

WRITABLE_ROOT=/vepfs-mlp2/queue013/public/huangborui
PROJECT_ROOT=$WRITABLE_ROOT/mani-project
PYTHON_BIN=$WRITABLE_ROOT/envs/stackcube_official/bin/python
TRAINER=$PROJECT_ROOT/third_party/ManiSkill-v3.0.1-official/ppo/ppo_fast.py
SOURCE=$PROJECT_ROOT/runs/two_robot_pick_cube_TRPC-O3_moderate_credit50m_seed0_20260723/ckpt_476.pt
LOG_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/$LOG_NAME
RUN_DIR=$PROJECT_ROOT/runs/$RUN_NAME

for path in "$LOG_DIR" "$RUN_DIR"; do
  case "$(realpath -m "$path")/" in
    "$WRITABLE_ROOT"/*) ;;
    *) echo "path escapes write boundary: $path" >&2; exit 91 ;;
  esac
done
[[ -f "$SOURCE" ]]
[[ ! -e "$LOG_DIR" ]]
[[ ! -e "$RUN_DIR" ]]
[[ "$PHYSICAL_GPU" == 0 || "$PHYSICAL_GPU" == 1 ]]

mkdir -p "$LOG_DIR" "$RUN_DIR"
IMMUTABLE_SOURCE=$RUN_DIR/source_o3_ckpt476.pt
cp "$SOURCE" "$IMMUTABLE_SOURCE"
sha256sum "$SOURCE" "$IMMUTABLE_SOURCE" >"$LOG_DIR/source_sha256.txt"

COMMAND=(
  "$PYTHON_BIN" "$TRAINER"
  --env-id TwoRobotPickCube-v1
  --seed 0
  --num-envs 1024
  --num-steps 100
  --update-epochs 8
  --num-minibatches 32
  --gamma 0.95
  --gae-lambda 0.95
  --learning-rate "$LEARNING_RATE"
  --anneal-lr
  --total-timesteps 10000000
  --num-eval-envs 16
  --num-eval-steps 100
  --cudagraphs
  --checkpoint "$IMMUTABLE_SOURCE"
  --exp-name "$RUN_NAME"
)

{
  echo "method_id=$METHOD_ID"
  echo "physical_gpu=$PHYSICAL_GPU"
  echo "learning_rate=$LEARNING_RATE"
  echo "additional_timesteps=10000000"
  echo "initialization=immutable copy of O3 ckpt_476"
  echo "started_at=$(date --iso-8601=seconds)"
  printf "command="
  printf "%q " "${COMMAND[@]}"
  printf "\n"
  "$PYTHON_BIN" --version
  sha256sum "$TRAINER"
} >"$LOG_DIR/launch_manifest.txt"
nvidia-smi -i "$PHYSICAL_GPU" -q >"$LOG_DIR/gpu_before.txt"

export XDG_CACHE_HOME=$WRITABLE_ROOT/runtime_cache
export TMPDIR=$WRITABLE_ROOT/runtime_tmp
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"

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
