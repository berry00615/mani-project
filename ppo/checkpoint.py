"""
Checkpoint save/load utilities.

Checkpoints are self-contained and cross-platform:
- No pickled environment objects
- No Linux-specific serialization
- Loadable on macOS with map_location="cpu" or map_location="mps"
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml


@dataclass
class CheckpointInfo:
    """Structured checkpoint metadata for inspection."""
    path: str = ""
    timestep: int = 0
    env_id: str = ""
    obs_mode: str = ""
    control_mode: str = ""
    obs_dim: int = 0
    act_dim: int = 0
    policy_hidden_sizes: list[int] = field(default_factory=list)
    value_hidden_sizes: list[int] = field(default_factory=list)
    action_low: Optional[list[float]] = None
    action_high: Optional[list[float]] = None
    versions: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def save_checkpoint(
    path: str,
    policy: torch.nn.Module,
    ppo: "PPO",
    timestep: int,
    env_id: str,
    obs_mode: str,
    control_mode: str,
    config: dict,
    rng_state: dict = None,
    obs_rms: dict = None,
    reward_rms: dict = None,
):
    """
    Save a training checkpoint.

    The checkpoint contains everything needed to resume training
    or load for evaluation/inspection on any platform.
    """
    import mani_skill
    import sapien

    os.makedirs(os.path.dirname(path), exist_ok=True)

    arch_info = policy.get_architecture_info()

    checkpoint = {
        # Model weights
        "policy_state_dict": policy.state_dict(),

        # Architecture (needed to rebuild the network)
        "architecture": arch_info,

        # Training state
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "timestep": timestep,

        # Environment info
        "env_id": env_id,
        "obs_mode": obs_mode,
        "control_mode": control_mode,

        # Config (training hyperparams)
        "config": config,

        # RNG state for resumable training
        "rng_state": rng_state or {},

        # Normalization stats (if used)
        "obs_rms": obs_rms,
        "reward_rms": reward_rms,

        # Software versions
        "versions": {
            "mani_skill": mani_skill.__version__,
            "sapien": sapien.__version__,
            "torch": torch.__version__,
        },
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
    device: torch.device = None,
) -> dict:
    """
    Load a checkpoint. Safe for cross-platform use.

    Args:
        path: Path to the .pt checkpoint file.
        map_location: torch map_location string (e.g. "cpu", "cuda:0").
        device: Optional torch device to move model weights to after loading.

    Returns:
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if device is not None:
        # Move policy state_dict to the target device
        for key in checkpoint["policy_state_dict"]:
            checkpoint["policy_state_dict"][key] = checkpoint["policy_state_dict"][key].to(device)

    return checkpoint


def get_checkpoint_info(path: str, map_location: str = "cpu") -> CheckpointInfo:
    """
    Read checkpoint metadata without creating an environment.

    Safe to call on any platform (Linux, macOS).
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    arch = checkpoint.get("architecture", {})

    # Convert numpy arrays to lists for safe dataclass storage
    action_low = arch.get("action_low", None)
    if hasattr(action_low, "tolist"):
        action_low = action_low.tolist()
    action_high = arch.get("action_high", None)
    if hasattr(action_high, "tolist"):
        action_high = action_high.tolist()

    return CheckpointInfo(
        path=path,
        timestep=checkpoint.get("timestep", 0),
        env_id=checkpoint.get("env_id", "unknown"),
        obs_mode=checkpoint.get("obs_mode", "unknown"),
        control_mode=checkpoint.get("control_mode", "unknown"),
        obs_dim=arch.get("obs_dim", 0),
        act_dim=arch.get("act_dim", 0),
        policy_hidden_sizes=arch.get("policy_hidden_sizes", []),
        value_hidden_sizes=arch.get("value_hidden_sizes", []),
        action_low=action_low,
        action_high=action_high,
        versions=checkpoint.get("versions", {}),
        config=checkpoint.get("config", {}),
    )
