#!/usr/bin/env bash
set -euo pipefail

WRITABLE_ROOT=/vepfs-mlp2/queue013/public/huangborui
PROJECT_ROOT=$WRITABLE_ROOT/mani-project
PYTHON_BIN=$WRITABLE_ROOT/envs/stackcube_official/bin/python
EVALUATOR=$PROJECT_ROOT/scripts/evaluate_official_two_robot_pick_cube.py
O4_RUN=$PROJECT_ROOT/runs/two_robot_pick_cube_TRPC-O4_o3ckpt476_lr1e4_10m_seed0_20260723
O5_RUN=$PROJECT_ROOT/runs/two_robot_pick_cube_TRPC-O5_o3ckpt476_lr5e5_10m_seed0_20260723
ROOT_OUT=$PROJECT_ROOT/logs/two_robot_pick_cube
O4_S=$ROOT_OUT/evaluation_o4_all_seed0_100env_20260724
O4_I=$ROOT_OUT/evaluation_o4_all_seed10000_100env_20260724
O5_S=$ROOT_OUT/evaluation_o5_all_seed0_100env_20260724
O5_I=$ROOT_OUT/evaluation_o5_all_seed10000_100env_20260724
FORMAL=$ROOT_OUT/evaluation_o4_o5_autobest_matchedseed20260723_1000env_20260724
PIPELINE=$ROOT_OUT/o4_o5_evaluation_pipeline_20260724

for path in "$O4_S" "$O4_I" "$O5_S" "$O5_I" "$FORMAL" "$PIPELINE"; do
  case "$(realpath -m "$path")/" in
    "$WRITABLE_ROOT"/*) ;;
    *) echo "path escapes write boundary: $path" >&2; exit 91 ;;
  esac
  [[ ! -e "$path" ]]
done
mkdir "$PIPELINE"

export XDG_CACHE_HOME=$WRITABLE_ROOT/runtime_cache
export TMPDIR=$WRITABLE_ROOT/runtime_tmp
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1
cd "$PROJECT_ROOT"

run_sweep() {
  local run_dir=$1
  local seed=$2
  local output=$3
  mkdir "$output"
  "$PYTHON_BIN" "$EVALUATOR" \
    --policy checkpoint --checkpoint-glob "$run_dir/*.pt" \
    --episodes 100 --seed "$seed" --horizon 100 \
    --output-json "$output/results.json" \
    --output-csv "$output/results.csv" \
    2>&1 | tee "$output/console.log"
}

run_sweep "$O4_RUN" 0 "$O4_S"
run_sweep "$O4_RUN" 10000 "$O4_I"
run_sweep "$O5_RUN" 0 "$O5_S"
run_sweep "$O5_RUN" 10000 "$O5_I"

BEST_CHECKPOINT=$(
  "$PYTHON_BIN" - \
    "$O4_S/results.json" "$O4_I/results.json" \
    "$O5_S/results.json" "$O5_I/results.json" <<'PY'
import json
import sys

pairs = ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4]))
ranked = []
for selection_path, independent_path in pairs:
    selection = {x["candidate"]: x for x in json.load(open(selection_path))}
    independent = {x["candidate"]: x for x in json.load(open(independent_path))}
    for path in sorted(set(selection) & set(independent)):
        s, i = selection[path], independent[path]
        ranked.append((
            s["successes"] + i["successes"],
            min(s["successes"], i["successes"]),
            s["mean_return"] + i["mean_return"],
            path,
        ))
ranked.sort(reverse=True)
if not ranked:
    raise SystemExit("no common checkpoints")
for row in ranked:
    print("\t".join(map(str, row)), file=sys.stderr)
print(ranked[0][-1])
PY
)
printf '%s\n' "$BEST_CHECKPOINT" >"$PIPELINE/auto_selected_checkpoint.txt"

mkdir "$FORMAL"
"$PYTHON_BIN" "$EVALUATOR" \
  --policy checkpoint --checkpoint-glob "$BEST_CHECKPOINT" \
  --episodes 1000 --seed 20260723 --horizon 100 \
  --output-json "$FORMAL/results.json" \
  --output-csv "$FORMAL/results.csv" \
  2>&1 | tee "$FORMAL/console.log"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "exit_status=0"
  echo "physical_gpu=1"
  echo "best_checkpoint=$BEST_CHECKPOINT"
} >"$PIPELINE/completion.txt"
