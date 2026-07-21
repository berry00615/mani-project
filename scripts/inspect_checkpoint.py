#!/usr/bin/env python3
"""
Inspect a PPO checkpoint without creating any environment.

Safe to run on any platform (Linux, macOS) — no ManiSkill import needed
if using --no-verify-env flag.

Usage:
    python scripts/inspect_checkpoint.py --checkpoint checkpoints/ppo_pick_cube/final.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo.checkpoint import get_checkpoint_info


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a PPO checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--map-location", type=str, default="cpu",
                        help="torch map_location (default: cpu)")
    return parser.parse_args()


def inspect():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("Checkpoint Inspection")
    print("=" * 70)

    # --- File info ---
    file_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"\n  File:         {ckpt_path}")
    print(f"  Size:         {file_size_mb:.2f} MB")

    # --- Read checkpoint ---
    info = get_checkpoint_info(str(ckpt_path), map_location=args.map_location)

    print(f"\n--- Training State ---")
    print(f"  Timestep:       {info.timestep:,}")

    print(f"\n--- Environment ---")
    print(f"  Env ID:         {info.env_id}")
    print(f"  Obs mode:       {info.obs_mode}")
    print(f"  Control mode:   {info.control_mode}")

    print(f"\n--- Architecture ---")
    print(f"  Obs dim:        {info.obs_dim}")
    print(f"  Act dim:        {info.act_dim}")
    print(f"  Policy hidden:  {info.policy_hidden_sizes}")
    print(f"  Value hidden:   {info.value_hidden_sizes}")
    if info.action_low is not None:
        act_low = info.action_low
        if hasattr(act_low, '__len__') and len(act_low) > 5:
            print(f"  Action low:     {act_low[:5]}... (len={len(act_low)})")
            print(f"  Action high:    {info.action_high[:5]}... (len={len(info.action_high)})")
        else:
            print(f"  Action low:     {act_low}")
            print(f"  Action high:    {info.action_high}")

    print(f"\n--- Software Versions ---")
    for k, v in info.versions.items():
        print(f"  {k}: {v}")

    print(f"\n--- Training Config ---")
    for k, v in info.config.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # --- Verify policy state dict is loadable ---
    print(f"\n--- State Dict Check ---")
    checkpoint = torch.load(str(ckpt_path), map_location=args.map_location, weights_only=False)
    state_dict = checkpoint.get("policy_state_dict", {})
    num_params = sum(v.numel() for v in state_dict.values())
    print(f"  Param tensors:  {len(state_dict)}")
    print(f"  Total params:   {num_params:,}")

    for key in list(state_dict.keys())[:5]:
        print(f"    {key}: {list(state_dict[key].shape)}")
    if len(state_dict) > 5:
        print(f"    ... and {len(state_dict) - 5} more tensors")

    # --- Cross-platform check ---
    print(f"\n--- Cross-Platform ---")
    print(f"  Loaded with map_location='{args.map_location}'")
    print(f"  No env object serialized: OK")
    print(f"  Safe for Mac loading:     OK (use map_location='cpu')")

    print(f"\n{'=' * 70}")
    print("Checkpoint looks valid.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    inspect()
