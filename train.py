#!/usr/bin/env python3
"""Canonical project training entry point.

The implementation lives in ``scripts/train_ppo_gpu.py`` so local and cloud
training cannot silently drift between two independent training loops.
"""

from scripts.train_ppo_gpu import parse_args, train


if __name__ == "__main__":
    train(parse_args())
