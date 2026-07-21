"""
Custom ManiSkill environments for the PPO project.

Importing this package registers custom environments with gymnasium.
"""

from .pick_cube_collision_penalty import PickCubeCollisionPenaltyEnv
from .pick_cube_collision_gripper import PickCubeCollisionGripperEnv
from .pick_cube_gripper_curriculum import PickCubeGripperCurriculumEnv
from .pick_cube_transport_curriculum import PickCubeTransportCurriculumEnv
from .pick_cube_lift_curriculum import PickCubeLiftCurriculumEnv

from .pick_cube_stable_lift_curriculum import PickCubeStableLiftCurriculumEnv

from .pick_cube_target_transport_curriculum import PickCubeTargetTransportCurriculumEnv
from .pick_cube_directed_transport_curriculum import PickCubeDirectedTransportCurriculumEnv

__all__ = [
    "PickCubeCollisionPenaltyEnv",
    "PickCubeCollisionGripperEnv",
    "PickCubeGripperCurriculumEnv",
    "PickCubeTransportCurriculumEnv",
    "PickCubeLiftCurriculumEnv",
    "PickCubeStableLiftCurriculumEnv",
    "PickCubeTargetTransportCurriculumEnv",
    "PickCubeDirectedTransportCurriculumEnv",
]
