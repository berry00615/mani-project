#!/usr/bin/env python3
"""Backward-compatible alias for the canonical GPU-capable trainer."""

from train_ppo_gpu import parse_args, train


if __name__ == "__main__":
    train(parse_args())
