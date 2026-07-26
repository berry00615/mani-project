#!/usr/bin/env bash
set -euo pipefail

WRITABLE_ROOT=/vepfs-mlp2/queue013/public/huangborui
PROJECT_ROOT=$WRITABLE_ROOT/mani-project
PYTHON_BIN=$WRITABLE_ROOT/envs/stackcube_official/bin/python
EVALUATOR=$PROJECT_ROOT/scripts/evaluate_official_two_robot_pick_cube.py
RUN_DIR=$PROJECT_ROOT/runs/two_robot_pick_cube_TRPC-O3_moderate_credit50m_seed0_20260723
TRAIN_LOG=$PROJECT_ROOT/logs/two_robot_pick_cube/training_TRPC-O3_moderate_credit50m_seed0_20260723
SELECTION_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/evaluation_o3_all_seed0_100env_20260723
INDEPENDENT_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/evaluation_o3_all_seed10000_100env_20260723
FORMAL_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/evaluation_o3_autobest_matchedseed20260723_1000env_20260723
PIPELINE_DIR=$PROJECT_ROOT/logs/two_robot_pick_cube/o3_posttrain_pipeline_20260723

for path in "$SELECTION_DIR" "$INDEPENDENT_DIR" "$FORMAL_DIR" "$PIPELINE_DIR"; do
  case "$(realpath -m "$path")/" in
    "$WRITABLE_ROOT"/*) ;;
    *) echo "path escapes write boundary: $path" >&2; exit 91 ;;
  esac
done

[[ ! -e "$SELECTION_DIR" ]]
[[ ! -e "$INDEPENDENT_DIR" ]]
[[ ! -e "$FORMAL_DIR" ]]
[[ ! -e "$PIPELINE_DIR" ]]
mkdir -p "$PIPELINE_DIR"

while [[ ! -f "$TRAIN_LOG/completion.txt" ]]; do
  date --iso-8601=seconds >>"$PIPELINE_DIR/wait_180s.log"
  sleep 180
done
grep -q '^exit_status=0$' "$TRAIN_LOG/completion.txt"

export XDG_CACHE_HOME=$WRITABLE_ROOT/runtime_cache
export TMPDIR=$WRITABLE_ROOT/runtime_tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
cd "$PROJECT_ROOT"

mkdir "$SELECTION_DIR"
"$PYTHON_BIN" "$EVALUATOR" \
  --policy checkpoint --checkpoint-glob "$RUN_DIR/*.pt" \
  --episodes 100 --seed 0 --horizon 100 \
  --output-json "$SELECTION_DIR/results.json" \
  --output-csv "$SELECTION_DIR/results.csv" \
  2>&1 | tee "$SELECTION_DIR/console.log"

mkdir "$INDEPENDENT_DIR"
"$PYTHON_BIN" "$EVALUATOR" \
  --policy checkpoint --checkpoint-glob "$RUN_DIR/*.pt" \
  --episodes 100 --seed 10000 --horizon 100 \
  --output-json "$INDEPENDENT_DIR/results.json" \
  --output-csv "$INDEPENDENT_DIR/results.csv" \
  2>&1 | tee "$INDEPENDENT_DIR/console.log"

BEST_CHECKPOINT=$(
  "$PYTHON_BIN" - "$SELECTION_DIR/results.json" "$INDEPENDENT_DIR/results.json" <<'PY'
import json
import sys

selection = {x["candidate"]: x for x in json.load(open(sys.argv[1]))}
independent = {x["candidate"]: x for x in json.load(open(sys.argv[2]))}
common = sorted(set(selection) & set(independent))
if not common:
    raise SystemExit("no common checkpoints")
best = max(
    common,
    key=lambda p: (
        selection[p]["successes"] + independent[p]["successes"],
        min(selection[p]["successes"], independent[p]["successes"]),
        selection[p]["mean_return"] + independent[p]["mean_return"],
    ),
)
print(best)
PY
)
printf '%s\n' "$BEST_CHECKPOINT" >"$PIPELINE_DIR/auto_selected_checkpoint.txt"

mkdir "$FORMAL_DIR"
"$PYTHON_BIN" "$EVALUATOR" \
  --policy checkpoint --checkpoint-glob "$BEST_CHECKPOINT" \
  --episodes 1000 --seed 20260723 --horizon 100 \
  --output-json "$FORMAL_DIR/results.json" \
  --output-csv "$FORMAL_DIR/results.csv" \
  2>&1 | tee "$FORMAL_DIR/console.log"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=0"
  echo "best_checkpoint=$BEST_CHECKPOINT"
} >"$PIPELINE_DIR/completion.txt"
