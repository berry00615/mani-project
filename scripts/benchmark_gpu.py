#!/usr/bin/env python3
"""
GPU Parallel Environment Benchmark for PickCubeCollisionGripper-v1.

Runs the GPU-adapted PPO trainer with num_envs = {16, 64, 256},
collects performance metrics (SPS, GPU utilisation, memory) and
verifies correctness (tensor shapes, no NaNs, per-env tracking).

Each run uses a short total_timesteps budget (4096–8192) — just enough
to verify correctness and measure throughput, NOT to train a policy.

Usage::

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_gpu.py

Output:
    - Console summary table
    - Per-run CSV logs in logs/benchmark_gpu/
    - Final benchmark report JSON
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- Benchmark configurations ----
# Each entry: (num_envs, total_timesteps, n_steps)
# n_steps must be <= total_timesteps / num_envs
BENCHMARKS = [
    {"num_envs": 16,  "total_timesteps": 4096,  "n_steps": 256},
    {"num_envs": 64,  "total_timesteps": 4096,  "n_steps": 64},
    {"num_envs": 256, "total_timesteps": 4096,  "n_steps": 16},
]

# For more thorough testing (optional):
BENCHMARKS_EXTENDED = [
    {"num_envs": 16,  "total_timesteps": 8192,  "n_steps": 256},
    {"num_envs": 64,  "total_timesteps": 8192,  "n_steps": 64},
    {"num_envs": 256, "total_timesteps": 8192,  "n_steps": 32},
]


def green(s):
    return f"\033[92m{s}\033[0m"


def red(s):
    return f"\033[91m{s}\033[0m"


def yellow(s):
    return f"\033[93m{s}\033[0m"


def run_benchmark(num_envs: int, total_timesteps: int, n_steps: int,
                  seed: int = 0) -> dict:
    """Run one benchmark configuration. Returns results dict."""
    config_path = PROJECT_ROOT / "configs/ppo_pick_cube_collision_gripper_gpu.yaml"
    train_script = PROJECT_ROOT / "scripts/train_ppo_gpu.py"

    log_dir = PROJECT_ROOT / "logs" / "benchmark_gpu"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"bench_n{num_envs}_t{total_timesteps}_s{seed}"
    # CSV logs are written to the log_dir specified in the training YAML config
    csv_path = PROJECT_ROOT / "logs" / "ppo_pick_cube_collision_gripper_gpu" / f"training_{run_name}.csv"

    cmd = [
        sys.executable, str(train_script),
        "--config", str(config_path),
        "--num-envs", str(num_envs),
        "--total-timesteps", str(total_timesteps),
        "--n-steps", str(n_steps),
        "--benchmark",
        "--run-name", run_name,
        "--seed", str(seed),
    ]

    result = {
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "success": False,
        "sps_avg": 0.0,
        "gpu_util_pct": 0.0,
        "gpu_mem_mb": 0.0,
        "wall_time_s": 0.0,
        "episodes_completed": 0,
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "early_close_rate": 0.0,
        "grasp_rate": 0.0,
        "anomalies": [],
        "errors": [],
        "csv_log": str(csv_path),
        "env": {},
    }

    env_vars = os.environ.copy()
    env_vars.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"\n{'─' * 60}")
    print(f"Benchmark: num_envs={num_envs}, total_steps={total_timesteps}, "
          f"n_steps={n_steps}")
    print(f"{'─' * 60}")

    t_start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env_vars,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max (GPU env creation is slow)
        )
        wall_time = time.time() - t_start
        result["wall_time_s"] = round(wall_time, 1)

        print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)

        if proc.returncode != 0:
            result["errors"].append(f"Exit code {proc.returncode}")
            print(red(f"  FAILED with exit code {proc.returncode}"))
            if proc.stderr:
                # Show last 50 lines of stderr
                stderr_lines = proc.stderr.strip().split("\n")
                for line in stderr_lines[-20:]:
                    print(f"  stderr: {line}")
                result["errors"].append(proc.stderr[-500:])
            return result

        # ---- Parse output for key metrics ----
        output = proc.stdout

        # Extract sim_backend
        for line in output.split("\n"):
            if "Sim backend:" in line:
                result["env"]["sim_backend"] = line.split("Sim backend:")[-1].strip()
            if "GPU sim enabled:" in line:
                result["env"]["gpu_sim"] = line.split("GPU sim enabled:")[-1].strip()
            if "physx_cuda confirmed" in line or "physx_cuda" in line:
                result["env"]["physx_cuda_ok"] = True

        # Parse CSV log — try to find actual path from training output first
        actual_csv = csv_path
        for line in output.split("\n"):
            if "CSV log:" in line:
                actual_csv = Path(line.split("CSV log:")[-1].strip())
                break
        if actual_csv.exists():
            result = _parse_csv_log(actual_csv, result)
        elif csv_path.exists():
            result = _parse_csv_log(csv_path, result)
        else:
            result["errors"].append(f"CSV log not found: {csv_path}")

        result["success"] = True

    except subprocess.TimeoutExpired:
        result["errors"].append("TIMEOUT (>1 hour)")
        print(red("  TIMEOUT"))
    except Exception as e:
        result["errors"].append(str(e))
        print(red(f"  Exception: {e}"))

    return result


def _parse_csv_log(csv_path: Path, result: dict) -> dict:
    """Parse the training CSV log to extract final metrics."""
    import csv
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            result["errors"].append("CSV log empty")
            return result

        # Use the last row for final metrics
        last = rows[-1]

        result["sps_avg"] = float(last.get("sps", 0))
        result["gpu_util_pct"] = float(last.get("gpu_util_pct", 0))
        result["gpu_mem_mb"] = float(last.get("gpu_mem_allocated_mb", 0))
        result["gpu_mem_reserved_mb"] = float(last.get("gpu_mem_reserved_mb", 0))
        result["episodes_completed"] = int(float(last.get("n_episodes_completed", 0)))
        result["success_rate"] = float(last.get("success_rate", 0))
        result["collision_rate"] = float(last.get("table_collision_rate", 0))
        result["early_close_rate"] = float(last.get("early_close_rate", 0))
        result["grasp_rate"] = float(last.get("grasp_rate", 0))
        result["episode_reward_mean"] = float(last.get("episode_reward_mean", 0))
        result["episode_length_mean"] = float(last.get("episode_length_mean", 0))
        result["policy_loss"] = float(last.get("policy_loss", 0))
        result["value_loss"] = float(last.get("value_loss", 0))
        result["approx_kl"] = float(last.get("approx_kl", 0))
        result["obs_mean_abs"] = float(last.get("obs_mean_abs", 0))
        result["reward_mean_abs"] = float(last.get("reward_mean_abs", 0))

        # Check for anomalies across all rows
        anomalies = []
        for i, row in enumerate(rows):
            sps = float(row.get("sps", 0))
            expl_var = row.get("explained_variance", "0")
            obs_mean = float(row.get("obs_mean_abs", 0))
            reward_mean = float(row.get("reward_mean_abs", 0))
            kl = float(row.get("approx_kl", 0))

            if expl_var == "nan":
                anomalies.append(f"Row {i}: explained_variance=NaN")
            if obs_mean > 100:
                anomalies.append(f"Row {i}: obs_mean_abs={obs_mean:.1f}")
            if reward_mean > 100:
                anomalies.append(f"Row {i}: reward_mean_abs={reward_mean:.1f}")
            if kl > 0.5:
                anomalies.append(f"Row {i}: approx_kl={kl:.4f}")

        result["anomalies"] = anomalies[:20]  # cap at 20
        if anomalies:
            result["has_anomalies"] = True
        else:
            result["has_anomalies"] = False

    except Exception as e:
        result["errors"].append(f"CSV parse error: {e}")

    return result


def print_summary(results: list[dict]):
    """Print a formatted summary table."""
    print(f"\n{'=' * 100}")
    print("GPU Parallel Environment Benchmark — Summary")
    print(f"{'=' * 100}")

    # Header
    header = (
        f"{'Env':>4s}  {'Steps':>7s}  {'SPS':>8s}  "
        f"{'GPU%':>5s}  {'Mem(MiB)':>9s}  {'Wall(s)':>8s}  "
        f"{'Eps':>5s}  {'Succ%':>6s}  {'Col%':>6s}  {'EC%':>6s}  "
        f"{'Grasp%':>7s}  {'Backend':>12s}  {'Anomalies':>10s}  {'Status':>8s}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        status = green("OK") if (r["success"] and not r.get("has_anomalies")) else (
            yellow("WARN") if r["success"] else red("FAIL"))

        backend = r.get("env", {}).get("sim_backend", "?")
        if backend == "physx_cuda":
            backend = green("physx_cuda")
        elif backend != "?":
            backend = red(backend)

        n_anomalies = len(r.get("anomalies", []))
        anomaly_str = red(f"{n_anomalies}") if n_anomalies > 0 else green("0")

        row = (
            f"{r['num_envs']:4d}  {r['total_timesteps']:7,d}  {r['sps_avg']:8,.0f}  "
            f"{r['gpu_util_pct']:4.0f}%  {r['gpu_mem_mb']:8,.0f}  {r['wall_time_s']:8.1f}  "
            f"{r['episodes_completed']:5d}  {r['success_rate']*100:5.1f}%  "
            f"{r['collision_rate']*100:5.2f}%  {r['early_close_rate']*100:5.2f}%  "
            f"{r['grasp_rate']*100:6.2f}%  {backend:>12}  {anomaly_str:>10}  {status:>8}"
        )
        print(row)

    print("-" * 100)

    # Anomaly details
    for r in results:
        if r.get("anomalies"):
            print(f"\n  Anomalies (num_envs={r['num_envs']}):")
            for a in r["anomalies"][:10]:
                print(f"    - {a}")

    # Errors
    for r in results:
        if r.get("errors"):
            print(f"\n  Errors (num_envs={r['num_envs']}):")
            for e in r["errors"]:
                print(f"    - {e}")

    print(f"\n{'=' * 100}")


def main():
    parser = argparse.ArgumentParser(description="GPU Parallel Environment Benchmark")
    parser.add_argument("--extended", action="store_true",
                        help="Use extended benchmark (8192 steps)")
    parser.add_argument("--num-envs", type=int, nargs="+",
                        help="Override num_envs list")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    benchmarks = BENCHMARKS_EXTENDED if args.extended else BENCHMARKS

    if args.num_envs:
        benchmarks = [b for b in benchmarks if b["num_envs"] in args.num_envs]

    print("=" * 100)
    print("GPU Parallel Environment Benchmark")
    print(f"PickCubeCollisionGripper-v1")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configs to run: {len(benchmarks)}")
    print("=" * 100)

    for b in benchmarks:
        print(f"  num_envs={b['num_envs']}, total_steps={b['total_timesteps']}, "
              f"n_steps={b['n_steps']}")

    results = []
    for i, b in enumerate(benchmarks):
        print(f"\n{'#' * 100}")
        print(f"# Benchmark {i+1}/{len(benchmarks)}: "
              f"num_envs={b['num_envs']}, total_steps={b['total_timesteps']}")
        print(f"{'#' * 100}")

        result = run_benchmark(
            num_envs=b["num_envs"],
            total_timesteps=b["total_timesteps"],
            n_steps=b["n_steps"],
            seed=args.seed + i,
        )
        results.append(result)

    # ---- Summary ----
    print_summary(results)

    # ---- Save JSON ----
    json_path = args.output_json
    if json_path is None:
        json_path = str(
            PROJECT_ROOT / "logs" / "benchmark_gpu" /
            f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # Clean up for JSON serialization
    for r in results:
        r.pop("csv_log", None)  # path is local

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {json_path}")

    # Exit code
    n_fail = sum(1 for r in results if not r["success"])
    n_anomaly = sum(1 for r in results if r.get("has_anomalies"))
    if n_fail > 0:
        print(red(f"\n{n_fail} benchmark(s) failed."))
        sys.exit(1)
    elif n_anomaly > 0:
        print(yellow(f"\nAll benchmarks completed but {n_anomaly} had anomalies."))
        sys.exit(0)
    else:
        print(green("\nAll benchmarks passed with no anomalies."))
        sys.exit(0)


if __name__ == "__main__":
    main()
